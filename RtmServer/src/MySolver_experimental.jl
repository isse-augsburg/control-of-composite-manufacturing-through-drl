module MySolver

export step2next!

using LinearAlgebra

function step!(
    t_step::Float64,
    t::Float64,
    iter::Int64,
    deltat::Float64,
    neutral_inds::Vector{Int64},

    # mesh properties
    N::Int64,
    celltype::Vector{Int64},
    num_neighbours,
    cellneighbours,
    cellneighboursarray,
    Amats,
    Tmats,
    cellfacearea,
    cellfacenormal,
    cellviscosity,
    cellpermeability,
    cellporosity,
    cellalpha,
    cellvolume,
    cellthickness,

    # simulation state
    p,
    gamma,
    u,
    v,
    rho,

    # additional parameters
    ap1,
    ap2,
    ap3,
    gamma_val,
    gamma_a, # boundary filling factor

    # preallocations
    Δp_old::Matrix{Float64},
    F_rho_num,
    F_u_num,
    F_v_num,
    F_gamma_num,
    F_gamma_num1, 
    uvec,
    bvec,
    temp,
    temp2,
    temp_mat
)
    for cell in range(1, N)

        # for all normal and all wall cells
        if celltype[cell] == 1 || celltype[cell] == -3
            
            # Adjust pressure gradient calculation to prevent backflow
            for i_neighbour in range(1, num_neighbours[cell])
                neighbour = cellneighboursarray[cell, i_neighbour]
                if celltype[neighbour] == -1 && p[neighbour] < p[cell]
                    bvec[i_neighbour] = 0.0
                else
                    bvec[i_neighbour] = p[neighbour] - p[cell]
                end
            end

            
            if num_neighbours[cell] == 3
                a = Amats[1, cell]
                b = Amats[2, cell]
                c = Amats[3, cell]
                d = Amats[4, cell]
                e = Amats[5, cell]
                f = Amats[6, cell]

                # Aplus=transpose(temp_mat) * temp_mat;
                a_ = a^2 + c^2 + e^2           
                b_ = a * b + c * d + e * f
                c_ = b_
                d_ = b^2 + d^2 + f^2
                
                b1 = a * bvec[1] + c * bvec[2] + e * bvec[3]
                b2 = b * bvec[1] + d * bvec[2] + f * bvec[3]
                inv = 1 / (a_ * d_ - b_ * c_)

                # 1 / (ad -bc) * [d -b; -c a]
                Δp_old[1, cell] = inv * d_ * b1 - inv * b_ * b2
                Δp_old[2, cell] = -inv * c_ * b1 + inv * a_ * b2

        
                # Δp_old[:, cell] .= temp_mat \ bvec  
            else
                a = Amats[1, cell]
                b = Amats[2, cell]
                c = Amats[3, cell]
                d = Amats[4, cell]
                # Δp_old[:, cell] .= Amat \ bvec[1:2]
                inv = 1 / (a * d - b * c)
                # inverse matrix 
                # 1 / (ad -bc) * [d -b; -c a]
                Δp_old[1, cell] = inv * d * bvec[1] - inv * b * bvec[2]
                Δp_old[2, cell] = -inv * c * bvec[1] + inv * a * bvec[2]
            end

            # solve linear system for pressure gradient
            # Save current Δp of this cell because it is needed to update rho, u, v, p, gamma (all for this timestep)


            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #   Calculations for each neighbour of the current cell.
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #
            # The following code loops once over the neighbours of the the current cell and calculates
            #
            # F_rho_num
            # F_u_num
            # F_v_num
            # F_gamma_num
            # F_gamma_num1
            #
            # simultaneously.

            F_rho_num[cell] = 0.0
            F_u_num[cell] = 0.0
            F_v_num[cell] = 0.0

            # gamma, volume of fluid method
            F_gamma_num[cell] = 0.0
            F_gamma_num1[cell] = 0.0

            for i_neighbour = range(1, num_neighbours[cell])
                neighbour::Int64 = cellneighboursarray[cell, i_neighbour] # index of this neighbour of the current cell

                if neighbour > 0
                    # cellfacearea of current cell, neighbour pairing
                    A::Float64 = cellfacearea[cell, i_neighbour]

                    if celltype[neighbour] == -1 # pressure inlet
                        # normalvelocity = -1 / cellviscosity[cell] * ([cellpermeability[cell] 0; 0 cellalpha[cell]*cellpermeability[cell]] * Δp_old[:, cell])' * [cellfacenormal[cell, i_neighbour, 1]; cellfacenormal[cell, i_neighbour, 2]]
                        
                        # normalvelocity = min(normalvelocity, 0)
                        # n_dot_rhou = normalvelocity * rho[neighbour]

                        # flow velocity

                        # uvec .= -1 / cellviscosity[cell] * [cellpermeability[cell] 0; 0 cellalpha[cell] * cellpermeability[cell]] * view(Δp_old, :, cell)
                        # effectively this happens
                        uvec[1] = -1 / cellviscosity[cell] * cellpermeability[cell] * Δp_old[1, cell]
                        uvec[2] = -1 / cellviscosity[cell] * cellalpha[cell] * cellpermeability[cell] * Δp_old[2, cell]

                        normalvelocity = uvec[1] * cellfacenormal[cell, i_neighbour, 1] + uvec[2] * cellfacenormal[cell, i_neighbour, 2]
                        normalvelocity = min(normalvelocity, 0)
                        n_dot_rhou = normalvelocity * rho[neighbour]

                        # rho
                        phi = 1
                        F_rho_num[cell] = F_rho_num[cell] + n_dot_rhou * phi * A

                        # u
                        phi = uvec[1]
                        F_u_num[cell] = F_u_num[cell] + n_dot_rhou * phi * A

                        # v
                        phi = uvec[2]
                        F_v_num[cell] = F_v_num[cell] + n_dot_rhou * phi * A

                        # gamma, volume of fluid method
                        normalvelocity = min(normalvelocity, 0) # must always be negative to avoid backflow
                        n_dot_u = normalvelocity
                        phi = gamma[neighbour]
                        F_gamma_num[cell] = F_gamma_num[cell] + n_dot_u * phi * A
                        phi = 1
                        F_gamma_num1[cell] = F_gamma_num1[cell] + n_dot_u * phi * A

                    else
                        # Calculation of u, v of the neighbour cell in the coordinate system of cell <cell>
                        # Transform vector [u[neighbour]; v[neighbour]] // x, y velocity values of neighbour cell in its own CS
                        # by multiplicating it with the precalculated transformation matrix 
                        # uvec .= Tmats[cell, i_neighbour, :, :] * [u[neighbour]; v[neighbour]]

                        uvec[1] = Tmats[cell, i_neighbour, 1, 1] * u[neighbour] + Tmats[cell, i_neighbour, 1, 2] * v[neighbour]
                        uvec[2] = Tmats[cell, i_neighbour, 2, 1] * u[neighbour] + Tmats[cell, i_neighbour, 2, 2] * v[neighbour]

                        u_cell = u[cell]
                        v_cell = v[cell]
                        u_neighbour = uvec[1]
                        v_neighbour = uvec[2]

                        n_x = cellfacenormal[cell, i_neighbour, 1]
                        n_y = cellfacenormal[cell, i_neighbour, 2]

                        # n dot (rho * [u; v])
                        # n_dot_rhou = [n_x; n_y]' * ((0.5 * (rho[cell] + rho[neighbour]) * [0.5 * (u_cell + u_neighbour); 0.5 * (v_cell + v_neighbour)]))
                        n_dot_rhou = 0.25 * (rho[cell] + rho[neighbour]) * (n_x * (u_cell + u_neighbour) + n_y * (v_cell + v_neighbour))

                        # rho
                        phi = 1
                        F_rho_num[cell] = F_rho_num[cell] + n_dot_rhou * phi * A


                        # u
                        if n_dot_rhou >= 0 # outflow
                            # first order upwinding
                            phi = u_cell
                        else
                            # first order upwinding
                            phi = u_neighbour
                        end

                        F_u_num[cell] = F_u_num[cell] + n_dot_rhou * phi * A

                        # v
                        if n_dot_rhou >= 0 # outflow
                            phi = v_cell
                        else
                            phi = v_neighbour
                        end

                        F_v_num[cell] = F_v_num[cell] + n_dot_rhou * phi * A

                        #  gamma, volume of fluid methods
                        # n_dot_u = [n_x; n_y]' * [0.5 * (u_cell + u_neighbour); 0.5 * (v_cell + v_neighbour)]
                        n_dot_u = n_x * 0.5 * (u_cell + u_neighbour) + n_y * 0.5 * (v_cell + v_neighbour)

                        if n_dot_u >= 0 # outflow
                            phi = gamma[cell]
                        else
                            phi = gamma[neighbour]
                        end

                        F_gamma_num[cell] = F_gamma_num[cell] + n_dot_u * phi * A
                        phi = 1
                        F_gamma_num1[cell] = F_gamma_num1[cell] + n_dot_u * phi * A
                    end
                end
            end
        end
    end
    # updates
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #   Calculation of rho, u, v, p, gamma updates.
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    
    for cell in range(1, N)
        if celltype[cell] != -1
            rho_temp = rho[cell]
            rho[cell] = clamp(rho_temp - deltat * (F_rho_num[cell] / cellvolume[cell]), 0., Inf)
            u[cell] = ((rho_temp * u[cell]) - deltat * (F_u_num[cell] / cellvolume[cell]) + deltat * (-Δp_old[1, cell])) / (deltat * cellviscosity[cell] / cellpermeability[cell] + rho[cell])
            v[cell] = ((rho_temp * v[cell]) - deltat * (F_v_num[cell] / cellvolume[cell]) + deltat * (-Δp_old[2, cell])) / (deltat * cellviscosity[cell] / (cellalpha[cell] * cellpermeability[cell]) + rho[cell])
            if gamma_val > 1.01
                # replace the ongoing interp1-call by the evaluation of
                # a polynomial for significant run-time reduction
                p[cell] = ap1 * rho[cell] ^ 2 + ap2  * rho[cell] + ap3
            else
                p[cell] = kappa * rho[cell] ^ gamma_val
            end
            gamma[cell] = clamp((cellporosity[cell] * gamma[cell] - deltat * (F_gamma_num[cell] - gamma[cell] * F_gamma_num1[cell]) / cellvolume[cell]) / cellporosity[cell], 0., 1.)
        else
            # boundary conditions
            u[cell] = 0 # u_a;
            v[cell] = 0 # v_a;
            rho[cell] = rho[cell]
            p[cell] = p[cell]
            gamma[cell] = gamma_a
        end
    end

    
    # adaptive timestepping
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #   adaptive time stepping
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if iter > 4
        weight_deltatnew = 0.5

        if gamma_val > 1.01
            betat2 = 0.1
        else
            betat2 = 0.05
        end

        temp[neutral_inds] = u[neutral_inds] 
        temp[neutral_inds] .^= 2
        temp2[neutral_inds] = v[neutral_inds] 
        temp2[neutral_inds] .^= 2
        temp .= temp + temp2
        temp .= sqrt.(temp)
        temp2[neutral_inds] = cellvolume[neutral_inds]
        temp2[neutral_inds] ./= cellthickness[neutral_inds]
        temp2 .= sqrt.(temp2) 
        temp2[neutral_inds] ./= temp[neutral_inds]

        deltat = (1 - weight_deltatnew) * deltat + weight_deltatnew * betat2 * minimum(temp2[neutral_inds])

        # Stabilitaetsbedingung/ CFL-Bedingung
        # Fluessigkeit darf innerhalb eines Zeitschritts nicht
        # weiter als 1 Zelle fliessen.
        # Dazu Schaeztung durch Strecke / Geschwindigkeit =
        # Zeit
        # wobei sqrt(Volumen/ Dicke) Schaetzung fuer Strecke
        # und betat2 ist ein Sicherheitsfaktor
        # min beachten, damit Stabilitaet global gewaehrleistet
        # ist.
        deltat = min(deltat, t_step / 10)
        if isnan(deltat)
            return
        end
    end

    # time update
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # time update
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    iter = iter + 1
    t = t + deltat

    return t,  iter,  deltat

end


function step2next!(
    t_next::Float64,
    t_step::Float64,
    t::Float64,
    t_max::Float64,
    iter::Int64,
    deltat::Float64,
    neutral_inds::Vector{Int64},

    # mesh properties
    N::Int64,
    celltype::Vector{Int64},
    num_neighbours,
    cellneighbours,
    cellneighboursarray,
    Amats,
    Tmats,
    cellfacearea,
    cellfacenormal,
    cellviscosity,
    cellpermeability,
    cellporosity,
    cellalpha,
    cellvolume,
    cellthickness,

    # simulation state
    p,
    gamma,
    u,
    v,
    rho,

    # additional parameters
    ap1,
    ap2,
    ap3,
    gamma_val,
    gamma_a::Float64, # boundary filling factor

    # preallocations for temporary values
    Δp_old,
    F_rho_num,
    F_u_num,
    F_v_num,
    F_gamma_num,
    F_gamma_num1,
    uvec,
    bvec,
    temp, 
    temp2, 
    temp_mat
)
    out = (0.::Float64, 0::Int64, 0.::Float64)
    # loop
    while t <= t_max
        if t >= t_next
            # return from function, but don't set finished, bc that would mean
            # that the max time is elapsed.
            finished = false
            return finished, t, iter, deltat
        end

       out = step!(
            t_step::Float64,
            t::Float64,
            iter::Int64,
            deltat::Float64,
            neutral_inds::Vector{Int64},

            # mesh properties
            N::Int64,
            celltype::Vector{Int64},
            num_neighbours::Vector{Int64},
            cellneighbours, #::Vector{Vector{Int64}},
            cellneighboursarray::Matrix{Int64},
            Amats::Matrix{Float64}, 
            Tmats, #::Array{Float64, 4},
            cellfacearea, #::Array{Float64, 3},
            cellfacenormal, #::MatrixArray{Float64, 3},
            cellviscosity::Vector{Float64},
            cellpermeability::Vector{Float64},
            cellporosity::Vector{Float64},
            cellalpha::Vector{Float64},
            cellvolume::Vector{Float64},
            cellthickness::Vector{Float64},

            # simulation state
            p::Vector{Float64},
            gamma::Vector{Float64},
            u::Vector{Float64},
            v::Vector{Float64},
            rho::Vector{Float64},

            # additional parameters
            ap1::Float64,
            ap2::Float64,
            ap3::Float64,
            gamma_val::Float64,
            gamma_a::Float64, # boundary filling factor

            # preallocations
            Δp_old,
            F_rho_num::Vector{Float64},
            F_u_num::Vector{Float64},
            F_v_num::Vector{Float64},
            F_gamma_num::Vector{Float64},
            F_gamma_num1::Vector{Float64},
            uvec::Vector{Float64},
            bvec::Vector{Float64},
            temp::Vector{Float64},
            temp2::Vector{Float64},
            temp_mat::Matrix{Float64}
        )
        t = out[1]
        iter = out[2]
        deltat = out[3]
    end
    
    # finished
    return true, t, iter, deltat
end

end

