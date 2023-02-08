import CairoMakie

const FIG = CairoMakie.Figure();

normal_dist(x::Real, μ::Real)          = normal_dist(x, μ, 1.0)
normal_dist(x::Real, μ::Real, σ::Real) = exp(-0.5 * abs2((x - μ) / σ)) / (σ * 2.5066282746310002)

target_1(x::Real) = ifelse(x < 0, 0.0, exp(-1.3 * x))

function demo!(fig::CairoMakie.Makie.Figure, target::Function, N::Int, init_range::Tuple{S,T}; xlims::Tuple{<:Real, <:Real}=(0.0, 6.0)) where {S<:Real, T<:Real}
    lb, rb = min(init_range...), max(init_range...)
    chain = Vector{promote_type(S,T)}(undef, N)
    @inbounds chain[1] = lb + rand() * (rb - lb)

    for t in 2:N
        this_state = @inbounds chain[t - 1]
        test_state = this_state + 1.0 * randn()
        this_target_value = target(this_state)
        test_target_value = target(test_state)
        Hastings_ratio = normal_dist(this_state, test_state) / normal_dist(test_state, this_state)
        accept_probility = Hastings_ratio * test_target_value / this_target_value
        @inbounds chain[t] = ifelse(rand() > accept_probility, this_state, test_state)
    end

    ref_x = collect(range(xlims...; length=501))
    ref_y = similar(ref_x)
    for i in eachindex(ref_y)
        @inbounds ref_y[i] = target(ref_x[i])
    end

    CairoMakie.empty!(fig)
    axis = @inbounds CairoMakie.Axis(fig[1,1])
    CairoMakie.hist!(axis, chain; bins=50, normalization=:pdf, color=(:gray, 0.5), strokewidth=0.5)
    CairoMakie.lines!(axis, ref_x, ref_y)
    return fig
end

demo!(FIG, target_1, 5000, (0.1, 2.0))

