module MCMCAnalysis

const VecI = AbstractVector
const MatI = AbstractMatrix

abstract type AbstractMCMCStrategy end

struct NoBurnInStrategy <: AbstractMCMCStrategy
    gen_num::UInt64  # num. of sampling generation
    param_a::Float64 # param. a of z ~ g(z) ∝ 1/√a, a ∈ [a⁻¹, a]

    NoBurnInStrategy(n::Integer)          = new(n, 2.0)
    NoBurnInStrategy(n::Integer, a::Real) = new(n, a)
end

struct BurnInStrategy <: AbstractMCMCStrategy
    gen_num::UInt64  # num. of sampling generation
    burn_in::UInt64  # num. of burn-in discarding stage
    param_a::Float64 # param. a of z ~ g(z) ∝ 1/√a, a ∈ [a⁻¹, a]

    BurnInStrategy(n::Integer, b::Integer) = new(n, b, 2.0)
    function BurnInStrategy(n::Integer, b::Integer, a::Real)
        b > 0 || error("BurnInStrategy: num. of burn-in is zero.")
        return new(n, b, a)
    end
end

abstract type AbstractMCMCSampler end
abstract type AbstractBurnInSampler <: AbstractMCMCSampler end

struct NoBurnInSampler <: AbstractBurnInSampler end
const NoBurnIn = NoBurnInSampler()

# type-stability ✓
for (c, p) in zip((:BurnInSampler, :RegularSampler), (:AbstractBurnInSampler, :AbstractMCMCSampler))
    @eval struct $c <: $p
        chain::Array{Float64,3}
        log_target::Array{Float64,2}

        $c(n::Integer, K::Integer, N::Integer) = new(Array{Float64,3}(undef, n, K, N), Array{Float64,2}(undef, K, N))
    end
end

struct MCMCSampler{
        SType <: AbstractMCMCStrategy,
        BType <: AbstractBurnInSampler
    } <: AbstractMCMCSampler
    strategy::SType
    burn_in_sampler::BType
    regular_sampler::RegularSampler

    # type-stability ✓
    MCMCSampler(
        strategy::SType,
        burn_in_sampler::BType,
        regular_sampler::RegularSampler
    ) where {
        SType <: AbstractMCMCStrategy,
        BType <: AbstractBurnInSampler
    } = new{SType,BType}(strategy, burn_in_sampler, regular_sampler)

    # type-stability ✓
    @generated function MCMCSampler(strategy::T, n::Int, K::Int) where T<:AbstractMCMCStrategy
        if T ≡ NoBurnInStrategy
            ex = :NoBurnIn
        elseif T ≡ BurnInStrategy
            ex = :(BurnInSampler(n, K, strategy.burn_in))
        end
        return :(MCMCSampler(strategy, $ex, RegularSampler(n, K, strategy.gen_num)))
    end
end

# = = = = = = = = = = = = = = = = = = = = = #
# Initialization of Sampler                 #
# = = = = = = = = = = = = = = = = = = = = = #

function initialize!(des::MatI, lb::NTuple{N,L}, ub::NTuple{N,U}) where {N,L<:Real,U<:Real}
    for k in axes(des, 2)
        @simd for i in axes(des, 1)
            @inbounds des[i,k] = lb[i] + rand() * (ub[i] - lb[i])
        end
    end
    return nothing
end

# type-stability ✓
@inline burn_in_init!(s::MCMCSampler{NoBurnInStrategy, NoBurnInSampler}, lb::L, ub::U) where {L,U} = nothing
@inline burn_in_init!(s::MCMCSampler{BurnInStrategy,   BurnInSampler},   lb::NTuple{N}, ub::NTuple{N}) where N =
    initialize!(view(s.burn_in_sampler.chain, :, :, 1), lb, ub)

# type-stability ✓
@inline regular_init!(s::MCMCSampler{NoBurnInStrategy, NoBurnInSampler}, log_target::F, n::I, one2K::O, lb::NTuple{N}, ub::NTuple{N}) where {F,I,O,N} =
    initialize!(view(s.regular_sampler.chain, :, :, 1), lb, ub)
@inline regular_init!(s::MCMCSampler{BurnInStrategy,   BurnInSampler},   log_target::Function, n::Int, one2K::Base.OneTo, lb::L, ub::U) where {L,U} = 
    evolve!(view(s.regular_sampler.chain, :, :, 1), view(s.burn_in_sampler.chain, :, :, s.strategy.burn_in), log_target, n, one2K, s.strategy.param_a)

# = = = = = = = = = = = = = = = = = = = = = #
# Subroutine: sampling                      #
# = = = = = = = = = = = = = = = = = = = = = #

# sampling from `iter` except for `exc`, type-stability ✓
function sampling(iter::AbstractUnitRange{T}, exc::T) where T
    ret = rand(iter)
    while ret ≡ exc
        ret = rand(iter)
    end
    return ret
end

# sampling of z ~ g(z) ∝ 1/√a, a ∈ [a⁻¹, a]
@inline sampling(a::Real)          = sampling(a, rand())
@inline sampling(a::Real, u::Real) = a * u * u - 2 * u * (u - 1) + abs2(u - 1) / a

# walker_new := the k-th walker's new vector
# walker_ole := the k-th walker's old vector
# walker_com := the randomly chosen walker's vector from complementary ensemble of the k-th walker
# type-stability ✓
function stretch_move!(walker_new::VecI, walker_old::VecI, walker_com::VecI, a::Real)
    z = sampling(a)
    @simd for i in eachindex(walker_new)
        @inbounds walker_new[i] = walker_com[i] + z * (walker_old[i] - walker_com[i])
    end
    return z
end

# type-stability ✓
@inline function log_accept_probability(log_target::Function, walker_new::VecI, walker_old::VecI, n::Int, z::Real)
    return (n - 1) * log(z) + log_target(walker_new) - log_target(walker_old)
end

# = = = = = = = = = = = = = = = = = = = = = #
# MCMC Evolution                            #
# = = = = = = = = = = = = = = = = = = = = = #

# type-stability ✓
function evolve!(walkers_new::MatI, walkers_old::MatI, log_target::Function, n::Int, one2K::Base.OneTo{Int}, a::Real)
    for k in one2K
        walker_old = view(walkers_old, :, k)
        walker_new = view(walkers_new, :, k)
        j = sampling(one2K, k)
        z = stretch_move!(walker_new, walker_old, view(walkers_old, :, j), a)
        q = log_accept_probability(log_target, walker_new, walker_old, n, z)

        if log(rand()) > q
            copyto!(walker_new, walker_old)
        end
    end
    return nothing
end

# type-stability ✓
function evolve!(s::AbstractMCMCSampler, log_target::Function, a::Real)
    chain = s.chain
    param_dim, epoch_num = size(chain, 1), size(chain, 3)
    one2K = axes(chain, 2)
    for t in 2:epoch_num
        evolve!(view(chain,:,:,t), view(chain,:,:,t-1), log_target, param_dim, one2K, a)
    end
    return nothing
end

end # module MCMCAnalysis
