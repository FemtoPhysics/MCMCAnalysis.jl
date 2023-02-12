module MCMCAnalysis

# sampling from `iter` except for `exc`
function sampling(iter::AbstractUnitRange{T}, exc::T) where T
    ret = rand(iter)
    while ret ≡ exc
        ret = rand(iter)
    end
    return ret
end

# sampling of z ~ g(z) ∝ 1/√a, a ∈ [a⁻¹, a]
sampling(a::Real)          = sampling(a, rand())
sampling(a::Real, u::Real) = a * u * u - 2 * u * (u - 1) + abs2(u - 1) / a

# walker_new := the k-th walker's new vector
# walker_ole := the k-th walker's old vector
# walker_com := the randomly chosen walker's vector from complementary ensemble of the k-th walker
function stretch_move!(walker_new::VecI, walker_old::VecI, walker_com::VecI, a::Real)
    z = sampling(a)
    @simd for i in eachindex(walker_new)
        @inbounds walker_new[i] = walker_com[i] + z * (walker_old[i] - walker_com[i])
    end
    return z
end

function log_accept_probability(log_target::Function, walker_new::VecI, walker_old::VecI, n::Int, z::Real)
    return (n - 1) * log(z) + log_target(walker_new) - log_target(walker_old)
end

end # module MCMCAnalysis
