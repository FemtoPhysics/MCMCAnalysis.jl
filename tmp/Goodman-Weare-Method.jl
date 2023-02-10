import CairoMakie

const VecI = AbstractVector

const FIG = CairoMakie.Figure()

target_1(x::Real) = ifelse(x < 0, 0.0, exp(-1.3 * x))

log_target_1(x::VecI{T}) where T<:Real = @inbounds ifelse(x[1] < 0, -Inf, -1.3 * x[1])

gsampling(a::Real)          = gsampling(a, rand())
gsampling(a::Real, u::Real) = a * u * u - 2 * u * (u - 1) + abs2(u - 1) / a

function demo_gsampling(; a::Real=2.0)
    sampling_z = Vector{Float64}(undef, 10000)
    for i in eachindex(sampling_z)
        @inbounds sampling_z[i] = gsampling(a)
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
    lb, rb = min(init_range...), max(init_range...)
    # chain of walkers:
    # axes(chain, 1) := walker's parameters, dims = n
    # axes(chain, 2) := number of walkers, dims = K
    # axes(chain, 3) := series of epochs, dims = N
    chain = Array{promote_type(S,T), 3}(undef, n, K, N)

    # Burn-in discarding
    this_state = view(chain, :, :, 1)
    for k in axes(this_state, 2), i in axes(this_state, 1)
        @inbounds this_state[i,k] = lb + rand() * (rb - lb)
    end
    let test_state = view(chain, :, :, 2)
        for _ in 1:burn_in_num
            for k in 1:K
                j = rand(1:K)
                while j ≡ K
                    j = rand(1:K)
                end
                z = gsampling(a)
                for i in 1:n
                    @inbounds test_state[i,k] = this_state[i,j] + z * (this_state[i,k] - this_state[i,j])
                end
                this_log_target_value = log_target(view(this_state, :, k))
                test_log_target_value = log_target(view(test_state, :, k))
                log_accept_probility  = (n-1) * log(1) + test_log_target_value - this_log_target_value
                if log(rand()) < log_accept_probility
                    for i in 1:n
                        @inbounds this_state[i,k] = test_state[i,k]
                    end
                end
            end
        end
    end

    for t in 2:N
        prev_state = view(chain, :, :, t-1)
        this_state = view(chain, :, :, t)
        for k in 1:K
            j = rand(1:K)
            while j ≡ K
                j = rand(1:K)
            end
            z = gsampling(a)
            for i in 1:n
                @inbounds this_state[i,k] = prev_state[i,j] + z * (prev_state[i,k] - prev_state[i,j])
            end
            prev_log_target_value = log_target(view(prev_state, :, k))
            this_log_target_value = log_target(view(this_state, :, k))
            log_accept_probility  = (n - 1) * log(1) + this_log_target_value - prev_log_target_value
            if log(rand()) > log_accept_probility
                for i in 1:n
                    @inbounds this_state[i,k] = prev_state[i,k]
                end
            end
        end
    end

    ref_x = collect(range(xlims...; length=501))
    ref_y = similar(ref_x)
    for i in eachindex(ref_y)
        @inbounds ref_y[i] = target(ref_x[i])
    end

    CairoMakie.empty!(fig)
    axis = @inbounds CairoMakie.Axis(fig[1,1])
    CairoMakie.hist!(axis, view(view(chain, 1, :, :), :);
                     bins=50, normalization=:pdf, color=(:gray, 0.5), strokewidth=0.5)
    CairoMakie.lines!(axis, ref_x, ref_y)
    return fig
end

demo_gsampling()
demo!(FIG, target_1, log_target_1, (0.1, 2.0); N=100, K=20, a=2.0, burn_in_num=100)
