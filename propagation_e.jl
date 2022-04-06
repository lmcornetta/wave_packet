#------------------------------------------------------------------------------------------------------------
#
#  Propagation of the vibrational wave packet
#  on the excited state
#
#------------------------------------------------------------------------------------------------------------

using Formatting
using FFTW

struct Param
    xmax::Float64
    xmin::Float64
    res::Int64
    dt::Float64
    timesteps::Int64
    dx::Float64
    x::Vector{Float64}
    dk::Float64
    k::Vector{Float64}
    im_time::Bool
    mass::Float64

    Param() = new(10.0,
                 -10.0,
                  512,
                  0.05,
                  1000,
                  2*10.0/512,
                  Vector{Float64}(-10.0+10.0/512 : 20.0/512 : 10.0),
                  pi/10.0,
                  Vector{Float64}(vcat(0:512/2-1, -512/2:-1)*pi/10.0),
                  false,
                  false,
                  1.0)
    
    Param(xmax::Float64, xmin::Float64, res::Int64, dt::Float64, timesteps::Int64, im_val::Bool, mass::Float64) = new(
        xmax, xmin, res, dt, timesteps, (xmax-xmin)/res, Vector{Float64}(xmin+0.5*(xmax-xmin)/res:(xmax-xmin)/res:xmax),
        2.0*pi/(xmax-xmin), Vector{Float64}(vcat(0:res/2-1, -res/2:-1)*2.0*pi/(xmax-xmin)), im_val, mass)
end

struct Excited_wfc
    res_x::Int64
    res_t::Int64
    grid::Vector{Vector{Complex{Float64}}}

    Excited_wfc(res_x::Int64, res_t::Int64) = new(res_x, res_t, [Vector{Complex{Float64}}(undef,res_t) for _ in 1:res_x])
end

mutable struct Operators
    V::Vector{Complex{Float64}}                 # Potential energy operator
    R::Vector{Complex{Float64}}                 # Coordinate space evolution operator exp(*potential energy*)
    K::Vector{Complex{Float64}}                 # Momentum space evolution operator exp(*kinetic energy*)
    wfc::Vector{Complex{Float64}}               # Wave function
    dump::Vector{Complex{Float64}}              # Dumping vector to avoid spurious reflexions at the end of the grid

    Operators(res) = new(zeros(res), zeros(res), zeros(res), zeros(res), ones(res))
end

function init(par::Param, v_::Vector{Complex{Float64}}, wfc_::Vector{Complex{Float64}})
    # @Function to initialize the wfc and potential instances
    #
    #   The last two boolean arguments control either the potential 
    #   and/or the wavefunction will be given as lists. Otherwise 
    #   they must be defined here.

    # @Initialize Operators instance
    opr = Operators(length(par.x))

    # @Initialize potential energy
    opr.V = v_

    # @Initialize wave function
    opr.wfc = wfc_
    density = abs2.(opr.wfc)
    norm2 = sum(density)*par.dx
    opr.wfc = opr.wfc./sqrt(norm2)

    # @Dumping
    dump_factor = 0.0
    icut = trunc(Int64,dump_factor*length(par.x))
    for i=1:icut
        opr.dump[i] = cos(0.5*pi*(par.x[i]-par.x[icut])/(par.xmin-par.x[icut]))
        opr.dump[length(par.x)-i+1] = cos(0.5*pi*(par.x[length(par.x)-i+1]-par.x[length(par.x)-icut+1])/(par.xmax-par.x[length(par.x)-icut+1]))
    end

    if (par.im_time)
        opr.K = exp.(-0.5*par.k.^2*par.dt/par.mass)
        opr.R = exp.(-0.5*opr.V*par.dt)
    else    
        opr.K = exp.(-im*0.5*par.k.^2*par.dt/par.mass)
        opr.R = exp.(-im*0.5*opr.V*par.dt)
    end

    return opr
end

# Calculation of (expectation value of) energy, <Psi|H|Psi>
function evaluate_energy(par, opr)
    # @Normalizing
    density = abs2.(opr.wfc)
    norm2 = sum(density)*par.dx
    wfc_r = opr.wfc./sqrt(norm2)

    # @Real, reciprocal and conjugate wfcs
    wfc_k = fft(wfc_r)
    wfc_c = conj(wfc_r)

    # @Finding the momentum and real-space terms
    energy_k = 0.5*wfc_c.*ifft((par.k.^2).*wfc_k)./par.mass
    energy_r = wfc_c.*opr.V.*wfc_r

    # @Integrating
    energy = real.(energy_k .+ energy_r)

    return sum(energy)*par.dx
end

#------------------------------------------------------------------------------------------------------------
function update_excited_wfc!(excited_wfc::Excited_wfc, wfc::Vector{Complex{Float64}}, iter::Int64)
    for i=1:length(wfc)
        excited_wfc.grid[i][iter] = wfc[i]
    end
end

#------------------------------------------------------------------------------------------------------------
# Split-operator integration
#
#   Basically, these are the functions that do all the fun.
#   They are divided in two functions: one without (split_op!) 
#   and one witho (split_op_animation!) animation gif of
#   the wave packet propagation.
#------------------------------------------------------------------------------------------------------------
function split_op!(par::Param, opr::Operators, excited_wfc::Excited_wfc)
    # Initial time
    t = 0.00

    # Initial density
    density = abs2.(opr.wfc)
    init_norm2 = sum(density)*par.dx

    for i = 1:par.timesteps
        # Updating excited_wfc object
        update_excited_wfc!(excited_wfc, opr.wfc, i)

        # Outputting data to file each 100 steps
        # Using the same interval (100 steps) for dumping corrections
        if ((i-1) % div(par.timesteps, 100) == 0)
            # Dump
            opr.wfc = opr.wfc .* opr.dump
            # Output files
            outfile = open("step"*string(i)*".dat","w")
            write(outfile, "#" * string(t) * "\n")
            for j = 1:length(density)
                write(outfile, string(par.x[j]) * "\t" *  string(real(opr.wfc[j])) * "\t" *  string(imag(opr.wfc[j])) * "\t" * string(density[j]) * "\n")
            end
            close(outfile)
        end

        # Half step in real space
        opr.wfc = opr.wfc .* opr.R
        # FFT to momentum space
        opr.wfc = fft(opr.wfc)
        # Full step in momentum space
        opr.wfc = opr.wfc .* opr.K
        # iFFT back to real space
        opr.wfc = ifft(opr.wfc)
        # Final half step in real space
        opr.wfc = opr.wfc .* opr.R
        # Density
        density = abs2.(opr.wfc)

        # Renormalizing for imaginary time
        if (par.im_time)
            renorm_factor = sum(density)*par.dx

            for j = 1:length(opr.wfc)
                opr.wfc[j] /= sqrt(renorm_factor)
            end
        end

        t += par.dt
    end
    println("Output successfully printed in .dat files ","\n")

    norm2 = sum(density)*par.dx
    printfmtln("Final wave packet norm squared: {:.5f}", norm2)    
    printfmtln("Norm loss index (fraction): {:.5f}\n", abs(init_norm2-norm2)/init_norm2)  
end


#------------------------------------------------------------------------------------------------------------

using QuadGK
using Interpolations

function psif0(par::Param, excited_wfc::Excited_wfc, omega::Float64, gamma::Float64)
    psif0 = Vector{Complex{Float64}}(zeros(par.res))
    tline = collect(range(0.0, par.dt*(par.timesteps - 1), step=par.dt))
    for i=1:par.res
        integrand = exp.((im*omega - gamma) .* tline) .* excited_wfc.grid[i]
    	# interp_linear_extrap = LinearInterpolation(tline, integrand, extrapolation_bc=Line())
    	interp_linear_extrap = LinearInterpolation(tline, integrand)
        psif0[i] = quadgk(interp_linear_extrap, 0, par.dt*(par.timesteps - 1), rtol=1e-3)[1]
	#psif0[i] = sum(exp.((im*omega - gamma) .* tline) .* excited_wfc.grid[i])*par.dt
    end
    density = abs2.(psif0)
    norm2 = sum(density)*par.dx
    return psif0 ./ sqrt(norm2)
end

#------------------------------------------------------------------------------------------------------------
function main(argv)
    # @Reading input and parameters from file
    #
    #   @1 Creating a Dict() object
    input_file = open(argv[1], "r")
    lines = read(input_file, String)
    params = Dict()
    for m in eachmatch(r"\s{0,10}(\w{2,15})\s{0,10}\:\s{0,10}(\w{1,15}\.?\w{1,10})", lines)
        key, value = m.captures
        params[lowercase(key)] = value
    end
    close(input_file)
    
    #   @2 Using the dictionary to set input variables
    dt = parse(Float64, params["dt"])
    timesteps = parse(Int64, params["timesteps"])
    mass = 1822.888486209*parse(Float64, params["mass"])
    im_val = parse(Bool, params["imaginary"])

    # The potential, wfc and xgrid vectors are read from the external files
    #       @2.a
    potential_c_file = open(params["potential"], "r")
    potential_c = Vector{Complex{Float64}}()
    while (!eof(potential_c_file))
        push!(potential_c, parse(Float64, readline(potential_c_file)))
    end
    potential_c = potential_c .- minimum(real.(potential_c))
	
    #       @2.b
    wfc_file = open(params["wfc"], "r")
    wfc = Vector{Complex{Float64}}()
    while (!eof(wfc_file))
        push!(wfc, parse(Float64, readline(wfc_file)))
    end

    #       @2.c
    xgrid_file = open(params["xgrid"], "r")
    xgrid = Vector{Float64}()
    while (!eof(xgrid_file))
        push!(xgrid, parse(Float64, readline(xgrid_file)))
    end

    res = length(potential_c)
    # Consistency check:
    res == length(wfc) ? nothing : throw(AssertionError("The potential and wavefunction grids must have the same dimension!\n"))
    res == length(xgrid) ? nothing : throw(AssertionError("The potential and x grids must have the same dimension!\n"))
    xmin = xgrid[1]
    xmax = xgrid[res]

    # @Starting parameters:
    #
    #   Parameters like xmax, xmin, grid size, time-step integration and,
    #   also, mass of the particle (or mode) go here. See initializator of
    #   struct Param for more information
    par = Param(xmax, xmin, res, dt, timesteps, im_val, mass)

    # @Starting operators:
    #
    #   Potential energy function, R and K operators and initial wave function 
    opr = init(par, potential_c, wfc)

    # @Main function
    println("=======================================================================================","\n")
    init_energy = evaluate_energy(par, opr)
    density = abs2.(opr.wfc)
    init_norm2 = sum(density)*par.dx
    printfmtln("Initial nuclear wave-packet built on the excited state. Initial energy: {:.5f}", init_energy)
    printfmtln("Initial wave-packet norm (before first dumping) squared: {:.5f}\n", init_norm2)

    # Creating and allocating memory for Excited_wfc object
    excited_wfc = Excited_wfc(par.res, par.timesteps)
    println("\t~ Excited state object successfully created ~ \n\t\t Memory allocated => ", excited_wfc.res_t*excited_wfc.res_x*sizeof(Complex{Float64}), " bytes\n")
    
    split_op!(par, opr, excited_wfc)

    final_energy = evaluate_energy(par, opr)
    printfmtln("Final energy: {:.5f}", final_energy)
    printfmtln("Total energy loss: {:.5f}\n", init_energy-final_energy)
    println("---------------------------------------------------------------------------------------\n\n")

    println("Creating Psif object")
    omega = 0.0
    gamma = 0.05/27.2107
    psif_initial = psif0(par, excited_wfc, omega, gamma)
    psif_density = abs2.(psif_initial)    

    psif0_file = open("psif0.dat","w")
    for j = 1:length(density)
        write(psif0_file, string(par.x[j]) * "\t" *  string(real(psif_initial[j])) * "\t" *  string(imag(psif_initial[j])) * "\t" * string(psif_density[j]) * "\n")
    end
    close(psif0_file)

    println("=======================================================================================")
end
#-----------------------------------------------------------------------------------------------------------

main(ARGS)
