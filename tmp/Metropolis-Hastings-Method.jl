import CairoMakie

const FIG = CairoMakie.Figure()

log_normal_dist(x::Real, μ::Real)          = -0.5 * abs2(x - μ) - 0.9189385332046727
log_normal_dist(x::Real, μ::Real, σ::Real) = -0.5 * abs2((x - μ) / σ) - log(σ) - 0.9189385332046727

target_1(x::Real) = ifelse(x < 0, 0.0, 1.3 * exp(-1.3 * x))
log_target_1(x::Real) = ifelse(x < 0, -Inf, 1.3 * (1 - x))

function demo!(
        fig::CairoMakie.Makie.Figure, target::Function, log_target::Function, N::Int, init_range::Tuple{S,T};
        burn_in_num::Int=50, xlims::Tuple{<:Real, <:Real}=(0.0, 6.0)
    ) where {S<:Real, T<:Real}
    lb, rb = min(init_range...), max(init_range...)
    chain = Vector{promote_type(S,T)}(undef, N)

    # Burn-in discarding
    this_state = lb + rand() * (rb - lb)
    test_state = zero(promote_type(S,T))
    for t in 1:burn_in_num
        test_state = this_state + 1.0 * randn()
        this_log_target_value = log_target(this_state)
        test_log_target_value = log_target(test_state)
        log_Hastings_ratio = log_normal_dist(this_state, test_state) - log_normal_dist(test_state, this_state)
        accept_probility = exp(log_Hastings_ratio + test_log_target_value - this_log_target_value)
        this_state = ifelse(rand() > accept_probility, this_state, test_state)
    end

    @inbounds chain[1] = this_state
    for t in 2:N
        this_state = @inbounds chain[t - 1]
        test_state = this_state + 1.0 * randn()
        this_log_target_value = log_target(this_state)
        test_log_target_value = log_target(test_state)
        log_Hastings_ratio = log_normal_dist(this_state, test_state) - log_normal_dist(test_state, this_state)
        accept_probility = exp(log_Hastings_ratio + test_log_target_value - this_log_target_value)
        @inbounds chain[t] = ifelse(rand() > accept_probility, this_state, test_state)
    end

    ref_x = collect(range(xlims...; length=501))
    ref_y = similar(ref_x)
    for i in eachindex(ref_y)
        @inbounds ref_y[i] = target(ref_x[i])
    end

    CairoMakie.empty!(fig)
    axis = @inbounds CairoMakie.Axis(fig[1,1])
    CairoMakie.hist!(axis, chain;
                     bins=50, normalization=:pdf, color=(:gray, 0.5), strokewidth=0.5)
    CairoMakie.lines!(axis, ref_x, ref_y)
    return fig
end

demo!(FIG, target_1, log_target_1, 5000, (0.1, 2.0))
