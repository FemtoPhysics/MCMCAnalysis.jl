import CairoMakie
import MCMCAnalysis: sampling, BurnInStrategy, MCMCSampler
import MCMCAnalysis: burn_in_init!, regular_init!, evolve!

const VecI = AbstractVector

const FIG = CairoMakie.Figure()

target_1(x::Real) = ifelse(x < 0, 0.0, 1.3 * exp(-1.3 * x))

log_target_1(x::VecI{T}) where T<:Real = @inbounds ifelse(x[1] < 0, -Inf, 1.3 * (1 - x[1]))

function demo_gsampling(; a::Real=2.0)
    sampling_z = Vector{Float64}(undef, 10000)
    for i in eachindex(sampling_z)
        @inbounds sampling_z[i] = sampling(a)
    end

    fig = CairoMakie.Figure()
    ax1 = @inbounds CairoMakie.Axis(fig[1,1])
    CairoMakie.hist!(ax1, sampling_z;
                     bins=500, normalization=:pdf, color=(:gray, 0.5), strokewidth=0.5)
    return fig
end

function demo!(
        fig::CairoMakie.Makie.Figure, target::Function, log_target::Function, init_range::Tuple{S,T};
        n::Int=1, N::Int=50, K::Int=20, a::Real=2.0, burn_in_num::Int=50, xlims::Tuple{<:Real, <:Real}=(0.0, 6.0)
    ) where {S<:Real, T<:Real}
    N < 2 && error("num. of epoch ($N) should be at least 2")

    lb, ub = (min(init_range...),), (max(init_range...),)
    handler = MCMCHandler(BurnInStrategy(n, K, N, burn_in_num, a))

    burn_in_init!(handler, lb, ub)
    evolve!(handler.burn_in_sampler, handler.strategy, log_target)

    regular_init!(handler, log_target, lb, ub)
    evolve!(handler.regular_sampler, handler.strategy, log_target)

    ref_x = collect(range(xlims...; length=501))
    ref_y = similar(ref_x)
    for i in eachindex(ref_y)
        @inbounds ref_y[i] = target(ref_x[i])
    end

    CairoMakie.empty!(fig)
    axis = @inbounds CairoMakie.Axis(fig[1,1])
    CairoMakie.hist!(axis, view(view(handler.regular_sampler.chain, 1, :, :), :);
                     bins=100, normalization=:pdf, color=(:gray, 0.3), strokewidth=0.5)
    CairoMakie.lines!(axis, ref_x, ref_y, linewidth=3.0)
    return fig
end

demo_gsampling()
demo!(FIG, target_1, log_target_1, (0.1, 2.0);
      N=200, K=32, a=2.0, burn_in_num=100, xlims=(0.0, 6.0))
