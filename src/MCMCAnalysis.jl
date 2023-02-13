module MCMCAnalysis

abstract type AbstractMCMCStrategy end

mutable struct NoBurnInStrategy <: AbstractMCMCStrategy
    gen_num::UInt64  # num. of sampling generation
    param_a::Float64 # param. a of z ~ g(z) ∝ 1/√a, a ∈ [a⁻¹, a]

    NoBurnInStrategy(n::Integer)          = new(n, 2.0)
    NoBurnInStrategy(n::Integer, a::Real) = new(n, a)
end

mutable struct BurnInStrategy <: AbstractMCMCStrategy
    gen_num::UInt64  # num. of sampling generation
    burn_in::UInt64  # num. of burn-in discarding stage
    param_a::Float64 # param. a of z ~ g(z) ∝ 1/√a, a ∈ [a⁻¹, a]

    BurnInStrategy(n::Integer, b::Integer) = new(n, b, 2.0)
    function BurnInStrategy(n::Integer, b::Integer, a::Real)
        b > 0 || error("BurnInStrategy: num. of burn-in is zero ($b).")
        return new(n, b, a)
    end
end

abstract type AbstractMCMCSampler end
abstract type AbstractBurnInSampler <: AbstractMCMCSampler end

struct NoBurnInSampler <: AbstractBurnInSampler end
const NoBurnIn = NoBurnInSampler()

struct BurnInSampler <: AbstractBurnInSampler
    chain::Array{Float64,3}
    log_target::Array{Float64,2}
end

struct RegularSampler <: AbstractMCMCSampler
    chain::Array{Float64,3}
    log_target::Array{Float64,2}
end

struct MCMCSampler{
        BurnInSamplerType <: AbstractBurnInSampler,
        MCMCStrategyType  <: AbstractMCMCStrategy
    } <: AbstractMCMCSampler
    burn_in_sampler::BurnInSamplerType
    regular_sampler::RegularSampler
    strategy::MCMCStrategyType

    function MCMCSampler(s::NoBurnInStrategy, n::Int, K::Int)
        return new{NoBurnInSampler, NoBurnInStrategy}(
            NoBurnIn, RegularSampler(
                Array{Float64,3}(undef, n, K, s.gen_num),
                Array{Float64,2}(undef, K, s.gen_num)
            ), s
        )
    end

    function MCMCSampler(s::BurnInStrategy, n::Int, K::Int)
        return new{BurnInSampler, BurnInStrategy}(
            BurnInSampler(
                Array{Float64,3}(undef, n, K, s.burn_in),
                Array{Float64,2}(undef, K, s.burn_in)
            ),
            RegularSampler(
                Array{Float64,3}(undef, n, K, s.gen_num),
                Array{Float64,2}(undef, K, s.gen_num)
            ), s
        )
    end
end

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
