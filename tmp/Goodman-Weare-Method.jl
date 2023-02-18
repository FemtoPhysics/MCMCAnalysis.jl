import CairoMakie
import MCMCAnalysis: sampling, BurnInStrategy, MCMCHandler
import MCMCAnalysis: burn_in_init!, regular_init!, evolve!

const VecI = AbstractVector

const FIG = CairoMakie.Figure()

target_1(x::Real)          = ifelse(x < 0, 0.0, 1.3 * exp(-1.3 * x))
target_2(x::Real, y::Real) = exp( -(100 * abs2(y - x * x) + abs2(1 - x)) / 20 )

log_target_1(x::VecI{T}) where T<:Real = @inbounds ifelse(x[1] < 0, -Inf, 1.3 * (1 - x[1]))
log_target_2(x::Real, y::Real)         = -(100 * abs2(y - x * x) + abs2(1 - x)) / 20
log_target_2(x::VecI{T}) where T<:Real = @inbounds log_target_2(x[1], x[2])

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
        ::Val{1}, fig::CairoMakie.Makie.Figure, target::Function, log_target::Function;
        gen_num::Int=50, walker_num::Int=20, a::Real=2.0, burn_in_num::Int=50,
        xlims::Tuple{<:Real, <:Real}=(0.0, 6.0)
    )
    gen_num < 2 && error("num. of epoch ($gen_num) should be at least 2")

    lb = (min(xlims...),)
    ub = (max(xlims...),)
    handler = MCMCHandler(BurnInStrategy(1, walker_num, gen_num, burn_in_num, a))

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
                     bins=200, normalization=:pdf, color=(:gray, 0.3), strokewidth=0.5)
    CairoMakie.lines!(axis, ref_x, ref_y, linewidth=3.0)
    return fig
end

function demo!(
        ::Val{2}, fig::CairoMakie.Makie.Figure, target::Function, log_target::Function;
        gen_num::Int=50, walker_num::Int=20, a::Real=2.0, burn_in_num::Int=50,
        xlims::Tuple{<:Real, <:Real}=(-20., 20.), ylims::Tuple{<:Real, <:Real}=(-20., 20.)
    )
    gen_num < 2 && error("num. of epoch ($gen_num) should be at least 2")

    lb = (min(xlims...), min(ylims...))
    ub = (max(xlims...), max(ylims...))
    handler = MCMCHandler(BurnInStrategy(2, walker_num, gen_num, burn_in_num, a))

    burn_in_init!(handler, lb, ub)
    evolve!(handler.burn_in_sampler, handler.strategy, log_target)

    regular_init!(handler, log_target, lb, ub)
    evolve!(handler.regular_sampler, handler.strategy, log_target)

    ref_x = collect(range(xlims...; length=501))
    ref_y = collect(range(ylims...; length=501))
    ref_z = Matrix{Float64}(undef, 501, 501)
    for j in eachindex(ref_y), i in eachindex(ref_x)
        @inbounds ref_z[i,j] = target(ref_x[i], ref_y[j])
    end

    CairoMakie.empty!(fig)
    axis = @inbounds CairoMakie.Axis(fig[1,1], limits=(xlims..., ylims...))
    CairoMakie.scatter!(axis,
        view(view(handler.regular_sampler.chain, 1, :, :), :),
        view(view(handler.regular_sampler.chain, 2, :, :), :);
        color=(:dodgerblue1, 0.005)
    )
    CairoMakie.contour!(axis, ref_x, ref_y, ref_z, linewidth=2.0, alpha=0.5)
    return fig
end

demo_gsampling()
demo!(Val(1), FIG, target_1, log_target_1; gen_num=800,  walker_num=32, a=2.0, burn_in_num=100, xlims=(0.0, 6.0))
demo!(Val(2), FIG, target_2, log_target_2; gen_num=3000, walker_num=32, a=2.0, burn_in_num=500, xlims=(-5., 6.), ylims=(-1.0, 36.))
