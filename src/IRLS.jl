__precompile__()

module IRLS

# Special Matrices
using Base.LinAlg: Diagonal, Hermitian, LowerTriangular, UpperTriangular
# Factorizations
using Base.LinAlg: qrfact, cholfact!
# Vector functions
using Base.LinAlg: norm

"""
    Distribution

Abstract type for distributions.
"""
abstract type Distribution end
"""
    Normal <: Distribution

Return a contrete type of the normal distribution.
"""
struct Normal <: Distribution end
"""
    Bernoulli <: Distribution

Return a contrete type of the Bernoulli distribution.
"""
struct Bernoulli <: Distribution end
"""
    Binomial <: Distribution

Return a contrete type of the Binomial distribution.
"""
struct Binomial <: Distribution end
"""
    Poisson <: Distribution

Return a contrete type of the Poisson distribution.
"""
struct Poisson <: Distribution end

"""
    AbstractLink

Abstract type for link functions.
"""
abstract type AbstractLink end
"""
    IdentityLink <: AbstractLink

Return a contrete type of the identity link function.
"""
struct IdentityLink <: AbstractLink end
"""
    LogitLink <: AbstractLink

Return a contrete type of the logit link function.
"""
struct LogitLink <: AbstractLink end
"""
    LogLink <: AbstractLink

Return a contrete type of the log link function.
"""
struct LogLink <: AbstractLink end

"""
    canonicallink(::Distribution)

Return the canonical link function for the distribution.
"""
canonicallink(obj::Distribution) = error("canonicallink is not defined for $(typeof(obj))")
canonicallink(obj::Normal) = IdentityLink()
canonicallink(obj::Union{Bernoulli,Binomial}) = LogitLink()
canonicallink(obj::Poisson) = LogLink()

"""
    logistic <: AbstractLink

Return a contrete type of the log link function.
"""
logistic(obj::Real) = inv(exp(-obj) + one(obj))

"""
    μμ′σ²(::Distribution, ::AbstractLink, ::AbstractVector{<:Real})
    μμ′σ²(::Distribution, ::AbstractLink, ::Real)

Return the value of the inverse link function, its derivative and the variance
of the distribution evaluated at the given value.
"""
μμ′σ²(::Distribution, ::AbstractLink, η::Real) = error("μμ′σ² is not defined for that combination of arguments.")

function μμ′σ²(distribution::Distribution, link::AbstractLink, η::AbstractVector{<:Real})
    m    = length(η)
    μ    = Vector{Float64}(m)
    μ′ = Vector{Float64}(m)
    σ²   = Vector{Float64}(m)
    @inbounds for idx ∈ eachindex(μ, μ′, σ², η)
        μ[idx], μ′[idx], σ²[idx] = μμ′σ²(distribution, link, η[idx])
    end
    return (μ, μ′, σ²)
end
function μμ′σ²(::Normal, ::IdentityLink, η::Real)
    return (η, one(Float64), one(Float64))
end
function μμ′σ²(::Union{Bernoulli,Binomial}, ::LogitLink, η::Real)
    g  = logistic(η)
    g′ = g * (one(g) - g)
    σ²  = g′
    return (g, g′, σ²)
end
function μμ′σ²(::Poisson, ::LogLink, η::Real)
    g  = exp(η)
    return (g, g, g)
end

function irls(A::AbstractMatrix{<:Real},
              b::AbstractVector{<:Real},
              distribution::Distribution;
              link::AbstractLink = canonicallink(distribution),
              maxit::Integer = 25)
    F = qrfact(A)
    Q = Matrix(F.Q)
    m, n = size(A)
    x = fill(zero(Float64), n)
    η = fill(zero(Float64), m)
    w = Vector{Float64}(undef, m)
    for itt ∈ 1:maxit
        μ, μ′, σ² = μμ′σ²(distribution, link, η)
        z  = η + (b - μ) ./ μ′
        w  = μ′.^2 ./ σ²
        x₀ = x
        C  = cholfact!(Hermitian(Q' * Diagonal(w) * Q)).factors
        x  = LowerTriangular(C') \ (Q' * Diagonal(w) * z)
        x  = UpperTriangular(C) \ x
        η  = Q * x
        norm(x - x₀, 2) < 1e-8 && return (F \ η, w)
    end
end
export
    irls
end
