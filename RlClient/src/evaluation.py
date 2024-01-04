from remote_rtm_env import RemoteRtmEnv
from pathlib import Path
import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

try:
    import imageio
except:
    pass

class Baseline:
    def __init__(self, action_type, num_inlets, nenvs, num_discrete_actions=5):
        self.action = np.ones((nenvs, num_inlets))
        if action_type == "discrete":
            self.action *= num_discrete_actions - 1

    def predict(self, imgs):
        return (self.action, None) # Match a real model's behaviour

class Evaluator:
    def __init__(self, server, reward_fn, inlets, action_type, num_discrete_actions=11, nenvs=1, use_fvc=True, use_pressure=True) -> None:
        self.env = RemoteRtmEnv(
            [server], 
            nenvs, 
            reward_fn, 
            "training",
            Path("/cfs/file_exchange"),
            use_fvc=use_fvc, 
            use_pressure=use_pressure, 
            action_type=action_type, 
            inlets=inlets, 
            num_discrete_actions=num_discrete_actions
        )
        self.env.reset()
        self.action_type = action_type
        self.num_discrete_actions = num_discrete_actions

    def eval_policy(self, model, file, save_path:Path, cut_baseline=False):
        inlets = self.env.inlets

        # format to reset env to file
        selects = [(1, str(file))]

        # EVALUATE MODEL

        # just reset, random file will be generated
        imgs = self.env.select_file(selects)

        # FVC MAP, will be saved later
        fvc_map = self.env.fvc_maps()[0]

        img = imgs[0, 0, :, :]
        model_imgs = [img]
        model_rewards = [0.]
        model_actions = [np.zeros(inlets)]

        done = False

        while not done:
            action = model.predict(imgs)

            if isinstance(action, tuple):
                action = action[0]

            imgs, reward, done, infos = self.env.step(action)
            img = imgs[0, 0, :, :]
            info = infos[0]

            if done:
                img = info["terminal_observation"][0, :, :]

            model_imgs.append(img)
            model_rewards.append(reward)
            model_actions.append(action[0, :])

        # EVALUATE BASELINE
        # just reset, random file will be generated
        imgs = self.env.select_file(selects)
        img = imgs[0, 0, :, :]
        base_imgs = [img]
        base_rewards = [0.]
        base_actions = [np.zeros(inlets)]

        done = False

        while not done:
            action = np.ones((1, inlets))

            if self.action_type == "discrete":
                action *= self.num_discrete_actions - 1

            imgs, reward, done, infos = self.env.step(action)
            img = imgs[0, 0, :, :]
            info = infos[0]

            if done:
                img = info["terminal_observation"][0, :, :]

            base_imgs.append(img)
            base_rewards.append(reward)
            base_actions.append(action[0, :])

        # SAVE
        print("Saving results...")
        save_path.mkdir(exist_ok=True, parents=True)
        raw_path = save_path / "raw"
        raw_path.mkdir(exist_ok=True, parents=True)

        fvc = Image.fromarray(fvc_map)
        fvc.save(raw_path / ("fvc_map.png"))
        print("FVC map done.")

        # make both list equally long
        if len(base_rewards) > len(model_rewards):
            model_rewards.extend([0. for _ in range(len(base_rewards) - len(model_rewards))])
        elif len(model_rewards) > len(base_rewards):
            base_rewards.extend([0. for _ in range(len(model_rewards) - len(base_rewards))])
        
        print(f"Rewards, base: {len(base_rewards)}, model: {len(model_rewards)}")

        rewards = np.array([base_rewards, model_rewards]).T
        df = pd.DataFrame(rewards, columns = ['base', 'model'], dtype=float)
        df.to_csv(save_path / ("rewards.csv"))
        print("Rewards done.")

        df_actions = pd.DataFrame(model_actions)
        df_actions.to_csv(save_path / ("model_actions.csv"))
        print("Actions done.")

        for i in range(max(len(model_imgs), len(base_imgs))):
            if i < len(base_imgs):
                base_img = Image.fromarray(base_imgs[i])
            else:
                # replicate last image
                base_img = Image.fromarray(base_imgs[-1])

            if i < len(model_imgs):
                model_img = Image.fromarray(model_imgs[i])
            elif not cut_baseline:
                # replicate last image
                model_img = Image.fromarray(model_imgs[-1])
            else: 
                break

            base_img.save(raw_path / (f"base00{i}.png"))
            model_img.save(raw_path / (f"model00{i}.png"))
        print("Images done.") 

    def eval_filelist(self, model, filelist:list, save_path:Path):
        nenvs = self.env.num_envs
        inlets = self.env.inlets

        selects = []
        current_files = []
        # format to reset env to file
        for i in range(1, nenvs + 1):
            file = filelist.pop()
            select = (i, str(file))
            selects.append(select)
            current_files.append(file)
            print(f"Setting env {i} to file: {file}")

        # EVALUATE MODEL
        imgs = self.env.select_file(selects)
        fvc_maps = self.env.fvc_maps()

        current_imgs = []
        current_actions = []
        current_rewards = []
        current_fvc = []
        for i in range(nenvs):
            img =  img = imgs[i, 0, :, :]
            current_imgs.append([img])
            current_actions.append([np.zeros(inlets)])
            current_rewards.append([0.])
            current_fvc.append(fvc_maps[i])

        env_finished = [False for _ in range(nenvs)]

        while not all(env_finished):
            actions = model.predict(imgs)

            if isinstance(actions, tuple):
                actions = actions[0]

            imgs, rewards, dones, infos = self.env.step(actions)
            
            selects = []
            for i in range(nenvs):
                if not env_finished[i]:
                    current_actions[i].append(actions[i, :])
                    current_rewards[i].append(rewards[i])

                    if dones[i]:
                        print(f"Env {i + 1} has finished.")
                        img = infos[i]["terminal_observation"][0, :, :]
                        current_imgs[i].append(img)

                        # save stuff
                        folder = save_path / current_files[i]
                        folder.mkdir(exist_ok=True, parents=True)
                        img_folder = folder / "raw"
                        img_folder.mkdir(exist_ok=True, parents=True)

                        for j, img in enumerate(current_imgs[i]):
                            image = Image.fromarray(img)
                            image.save(img_folder / (f"model00{j}.png"))

                        fvc_image = Image.fromarray(current_fvc[i])
                        fvc_image.save(img_folder / (f"fvc_map.png"))

                        df = pd.DataFrame(current_rewards[i], columns = ['reward'], dtype=float)
                        df.to_csv(folder / ("rewards.csv"))

                        df = pd.DataFrame(current_actions[i])
                        df.to_csv(folder / ("model_actions.csv"))

                        # reset lists

                        try:
                            file = filelist.pop()
                            select = (i + 1, str(file))
                            selects.append(select)
                            current_files[i] = file
                            print(f"Setting env {i + 1} to file: {file}")
                            
                        except IndexError:
                            print(f"No files left. Env {i + 1} will be ignored from now on.")
                            env_finished[i] = True

                    else:
                        img = imgs[i, 0, :, :]
                        current_imgs[i].append(img)

            if len(selects) > 0:
                imgs_tmp = self.env.select_file(selects)
                fvc_maps = self.env.fvc_maps()

                for j, select in enumerate(selects):
                    env_id = select[0] - 1
                    img = imgs_tmp[j, 0, :, :]
                    fvc_img = fvc_maps[env_id]

                    current_imgs[env_id] = [img]
                    current_fvc[env_id] = fvc_img
                    current_actions[env_id] = [np.zeros(inlets)]
                    current_rewards[env_id] = [0.]

                    imgs[env_id] = imgs_tmp[j, 0, :, :]


def fancy_plot(save_path: Path, inlets=5, action_type="box", num_discrete_actions=3):
    plot_path = save_path / "plots"
    plot_path.mkdir(exist_ok=True, parents=True)

    # load model actions
    df = pd.read_csv(save_path / "model_actions.csv")
    model_actions = df.to_numpy()
    inlet_bar = 48
    base_action = np.ones((inlet_bar, 5))
    model_action = np.ones((inlet_bar, 5))

    if action_type == "discrete":
        model_actions /= num_discrete_actions - 1
    elif action_type == "box":
        model_actions = model_actions * 0.5 + 0.5
    
    # load rewards 
    df = pd.read_csv(save_path / "rewards.csv")
    rewards = df.to_numpy()
    model_rewards = rewards[:, 2]
    base_rewards = rewards[:, 1]
    n_rewards = base_rewards.shape[0]
    n_actions = model_actions.shape[0]

    # remove overlapping rewards
    for i in range(1, n_rewards):
        if base_rewards[i] == 0. and np.all(np.where(base_rewards[i:] == 0., True, False)):
            base_rewards = base_rewards[:i]
            break
        if model_rewards[i] == 0. and np.all(np.where(model_rewards[i:] == 0., True, False)):
                model_rewards = model_rewards[:i]
                break

    model_acc = []
    base_acc = []

    filenames = []
    for i in range(n_rewards):
        fig, ax = plt.subplots(2, 4, gridspec_kw={'width_ratios': [1, 21, 21, 21], 'height_ratios': [1, 1]})

        fig.set_size_inches(16, 8)
        
        b_action = ax[0, 0]
        b_img = ax[0, 1]

        rew = ax[0, 2]
        acc = ax[1, 2]

        m_action = ax[1, 0]
        m_img = ax[1, 1]

        model_action_plot = ax[1, 3]
        base_action_plot = ax[0, 3]

        axis = [m_action, m_img, b_action, b_img]
        for a in axis:
            a.axis('off')

        with Image.open(save_path / "raw" / f"model00{i}.png") as img:
            model_img = np.asarray(img)

        with Image.open(save_path / "raw" / f"base00{i}.png") as img:
            base_img = np.asarray(img)

        # create model action image
        # cap index in case baseline took longer than model
        # action will be set to zero
        if i >= n_actions:
            m = np.zeros(inlets)
        else:
            m = model_actions[i, 1:]

        step = int(inlet_bar / inlets)
        for j in range(inlets):
            model_action[j * step: (j + 1) * step, :] = m[j]

        max_p = 5.
        min_p = 0.1
        for j in range(inlets):
            a = (model_actions[:i + 1, j + 1]) * (max_p - min_p) + min_p
            model_action_plot.plot(a, label=f"gate {j + 1}")
            base_action_plot.plot(np.ones(i + 1) * max_p, label=f"gate {j + 1}")
            
        model_action_plot.set_xlim([0, n_rewards])
        model_action_plot.set_ylim([0., 7.5])
        model_action_plot.set_ylabel("pressure in bar")
        model_action_plot.set_title("model action/step")
        model_action_plot.legend()

        base_action_plot.set_xlim([0, n_rewards])
        base_action_plot.set_ylim([0., 7.5])
        base_action_plot.set_ylabel("pressure in bar")
        base_action_plot.set_title("baseline action/step")
        base_action_plot.legend()
        

        p1 = b_action.imshow(base_action)
        p1.set_clim(0., 1.)
        p2 = b_img.imshow(base_img)
        p2.set_clim(0, 255)
        p3 = m_action.imshow(model_action)
        p3.set_clim(0., 1)
        p4 = m_img.imshow(model_img)
        p4.set_clim(0, 255)

        b_img.set_title("baseline")
        m_img.set_title("model")

        rew.plot(model_rewards[:i + 1], label="model")
        rew.plot(base_rewards[:i + 1], label="base")
        rew.set_xlim([0, n_rewards])
        rew.set_ylim([-50, 150])
        rew.set_title("reward/step")
        rew.legend()

        model_acc.append(np.sum(model_rewards[:i + 1]))
        base_acc.append(np.sum(base_rewards[:i + 1]))

        acc.plot(model_acc, label="model")
        acc.plot(base_acc, label="base")
        acc.set_xlim([0, n_rewards])
        acc.set_ylim([-500, 50])
        acc.set_title("accumulated reward")
        acc.legend()
        
        plt.tight_layout()
        f = plot_path / f"plot0{i}.png"
        filenames.append(f)
        plt.savefig(f)
        # plt.show()
        plt.close(fig)

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(save_path / 'comparison.gif', images, fps=2)

def stats(folders:list):
    results = {}

    # experiment folders
    for folder in folders:
        ep_folders = os.listdir(folder)
        ep_n = len(ep_folders)

        # save accumulated rewards and lengths per episode (and for baseline/ model)
        ep_acc_rew = np.zeros(ep_n)
        ep_len = np.zeros(ep_n)
        ep_step_rew = np.zeros(ep_n)

        # read rewards.csv from all subfolders
        for i, f in enumerate(ep_folders):
            if not os.path.isdir(folder / f):
                continue
            df = pd.read_csv(folder/ f / "rewards.csv")
            model_rewards = df.to_numpy()[:, 1]

            ep_acc_rew[i] = np.sum(model_rewards)

            ep_len[i] = model_rewards.shape[0]    

            ep_step_rew[i] = ep_acc_rew[i] / ep_len[i] 

        
        model_mean_rew = np.mean(ep_acc_rew)
        model_mean_len = np.mean(ep_len)

        name = folder.parts[-1] + "-" + folder.parts[-2]
        results[name] = (model_mean_rew, model_mean_len)

    return results

def ep_len_csv(save_path:Path):
    data = {}
    for i in range(1, 101):
        f = f"mesh{i}.jld2"
        file = save_path / f / "rewards.csv"
        df = pd.read_csv(file)
        rewards = df.to_numpy()[:, 1]
        ep_len = rewards.shape[0]
        data[f] = ep_len

    df = pd.DataFrame.from_dict(data, "index")
    df.to_csv(save_path / "ep_lens.csv")


def plot_comparison(save_path:Path, baseline:Path, inlets=3, action_type="box", num_discrete_actions=3):
    ep_folders = os.listdir(baseline)
    ep_n = len(ep_folders)

    for i, f in enumerate(ep_folders):
        if not os.path.isdir(save_path / f):
            continue
        if os.path.exists(save_path / f/ "comparison.gif"):
            continue
        plot_path = save_path / f / "plots"
        plot_path.mkdir(exist_ok=True, parents=True)

        # load model actions
        df = pd.read_csv(save_path / f / "model_actions.csv")
        model_actions = df.to_numpy()
        inlet_bar = 48
        base_action = np.ones((inlet_bar, 5))
        model_action = np.ones((inlet_bar, 5))

        if action_type == "discrete":
            model_actions /= num_discrete_actions - 1
        elif action_type == "box":
            model_actions = model_actions * 0.5 + 0.5
        
        # load rewards 
        df = pd.read_csv(save_path / f / "rewards.csv")
        model_rewards = df.to_numpy()[:, 1]
        df = pd.read_csv(baseline / f / "rewards.csv")
        base_rewards = df.to_numpy()[:, 1]

        n_model = model_rewards.shape[0]
        n_base = base_rewards.shape[0]

        model_acc = []
        base_acc = []

        filenames = []
        for i in range(n_model):
            fig, ax = plt.subplots(2, 4, gridspec_kw={'width_ratios': [1, 21, 21, 21], 'height_ratios': [1, 1]})

            fig.set_size_inches(16, 8)
            
            b_action = ax[0, 0]
            b_img = ax[0, 1]

            rew = ax[0, 2]
            acc = ax[1, 2]

            m_action = ax[1, 0]
            m_img = ax[1, 1]

            model_action_plot = ax[1, 3]
            base_action_plot = ax[0, 3]

            axis = [m_action, m_img, b_action, b_img]
            for a in axis:
                a.axis('off')

            with Image.open(save_path / f / "raw" / f"model00{i}.png") as img:
                model_img = np.asarray(img)

            if i < n_base:
                with Image.open(baseline / f  / "raw" / f"model00{i}.png") as img:
                    base_img = np.asarray(img)
            else:
                base_img = np.zeros((50, 50), dtype=np.uint8)

            # create model action image
            # cap index in case baseline took longer than model
            # action will be set to zero
            if i >= n_model:
                m = np.zeros(inlets)
            else:
                m = model_actions[i, 1:]

            step = int(inlet_bar / inlets)
            for j in range(inlets):
                model_action[j * step: (j + 1) * step, :] = m[j]

            max_p = 5.
            min_p = 0.1
            for j in range(inlets):
                a = (model_actions[:i + 1, j + 1]) * (max_p - min_p) + min_p
                model_action_plot.plot(a, label=f"gate {j + 1}")
                base_action_plot.plot(np.ones(i + 1) * max_p, label=f"gate {j + 1}")
                
            model_action_plot.set_xlim([0, n_model])
            model_action_plot.set_ylim([0., 7.5])
            model_action_plot.set_ylabel("pressure in bar")
            model_action_plot.set_title("model action/step")
            model_action_plot.legend()

            base_action_plot.set_xlim([0, n_model])
            base_action_plot.set_ylim([0., 7.5])
            base_action_plot.set_ylabel("pressure in bar")
            base_action_plot.set_title("baseline action/step")
            base_action_plot.legend()
            

            p1 = b_action.imshow(base_action)
            p1.set_clim(0., 1.)
            p2 = b_img.imshow(base_img)
            p2.set_clim(0, 255)
            p3 = m_action.imshow(model_action)
            p3.set_clim(0., 1)
            p4 = m_img.imshow(model_img)
            p4.set_clim(0, 255)

            b_img.set_title("baseline")
            m_img.set_title("model")

            rew.plot(model_rewards[:i + 1], label="model")
            rew.plot(base_rewards[:i + 1], label="base")
            rew.set_xlim([0, n_model])
            rew.set_ylim([-50, 150])
            rew.set_title("reward/step")
            rew.legend()

            model_acc.append(np.sum(model_rewards[:i + 1]))
            base_acc.append(np.sum(base_rewards[:i + 1]))

            acc.plot(model_acc, label="model")
            acc.plot(base_acc, label="base")
            acc.set_xlim([0, n_model])
            acc.set_ylim([-500, 50])
            acc.set_title("accumulated reward")
            acc.legend()
            
            plt.tight_layout()
            save_img = plot_path / f"plot0{i}.png"
            filenames.append(save_img)
            plt.savefig(save_img)
            # plt.show()
            plt.close(fig)

        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(save_path  / f / 'comparison.gif', images, fps=2)
        print(f"{save_path / f} done.")

def plot_model(save_path, inlets=3, action_type="box", num_discrete_actions=3): 
    plot_path = save_path / "plots"
    plot_path.mkdir(exist_ok=True, parents=True)

    # load model actions
    df = pd.read_csv(save_path / "model_actions.csv")
    model_actions = df.to_numpy()
    inlet_bar = 48
    model_action = np.ones((inlet_bar, 5))

    if action_type == "discrete":
        model_actions /= num_discrete_actions - 1
    elif action_type == "box":
        model_actions = model_actions * 0.5 + 0.5
    
    # load rewards 
    df = pd.read_csv(save_path / "rewards.csv")
    rewards = df.to_numpy()
    model_rewards = rewards[:, 1]
    n_rewards = model_rewards.shape[0]
    n_actions = model_actions.shape[0]

    model_acc = []

    filenames = []
    for i in range(n_rewards):
        fig, ax = plt.subplots(1, 4, gridspec_kw={'width_ratios': [1, 21, 21, 21]})

        fig.set_size_inches(16, 4)

        acc = ax[2]

        m_action = ax[0]
        m_img = ax[1]

        model_action_plot = ax[3]

        axis = [m_action, m_img]
        for a in axis:
            a.axis('off')

        with Image.open(save_path / "raw" / f"model00{i}.png") as img:
            model_img = np.asarray(img)

        m = model_actions[i, 1:]

        step = int(inlet_bar / inlets)
        for j in range(inlets):
            model_action[j * step: (j + 1) * step, :] = m[j]

        max_p = 5.
        min_p = 0.1
        for j in range(inlets):
            a = (model_actions[:i + 1, j + 1]) * (max_p - min_p) + min_p
            model_action_plot.plot(a, label=f"gate {j + 1}")
            
        model_action_plot.set_xlim([0, n_rewards])
        model_action_plot.set_ylim([0., 7.5])
        model_action_plot.set_ylabel("pressure in bar")
        model_action_plot.set_title("model action/step")
        model_action_plot.legend()

        p3 = m_action.imshow(model_action)
        p3.set_clim(0., 1)
        p4 = m_img.imshow(model_img)
        p4.set_clim(0, 255)

        m_img.set_title("model")

        model_acc.append(np.sum(model_rewards[:i + 1]))

        acc.plot(model_acc)
        acc.set_xlim([0, n_rewards])
        acc.set_ylim([-500, 50])
        acc.set_title("accumulated reward")
        acc.legend()
        
        plt.tight_layout()
        f = plot_path / f"plot0{i}.png"
        filenames.append(f)
        plt.savefig(f)
        # plt.show()
        plt.close(fig)

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(save_path / 'comparison.gif', images, fps=2)



        

        
        

        