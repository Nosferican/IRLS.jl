__precompile__()

module IRLS

using LinearAlgebra: BlasReal, norm, qrfact, cholfact!, Hermitian, LowerTriangular, UpperTriangular

abstract type Distribution end
struct Normal <: Distribution end
struct Bernoulli <: Distribution end
struct Poisson <: Distribution end

abstract type AbstractLink end
struct IdentityLink <: AbstractLink end
struct LogitLink <: AbstractLink end
struct LogLink <: AbstractLink end

logistic(obj::Real) = inv(exp(-obj) + one(obj))

linkinv(::IdentityLink, η::BlasReal) = η
linkinv(::LogitLink, η::BlasReal) = logistic(η)
linkinv(::LogLink, η::BlasReal) = exp(η)

function ∂μ∂η(::LogitLink, η::Real)
    output = logistic(η)
    output *= (one(output) - output)
    return output
end

"""
    μ∂μ∂ησ(::Distribution, ::AbstractLink, ::Real)

Return the value of the inverse link function, its derivative
and the standard deviation of the distribution evaluated at the
given value.
"""
function μ∂μ∂ησ(::Distribution, ::AbstractLink, η::Real)
end

function μ∂μ∂ησ(::Normal, ::IdentityLink, η::Real)
    return (η, one(Float64), 1)
end
function μ∂μ∂ησ(::Bernoulli, ::LogitLink, η::Real)
    g  = logistic(η)
    g′ = g * (one(g) - g)
    σ  = sqrt(η * (one(η) - η))
    return (g, g′, σ)
end
function μ∂μ∂ησ(::Poisson, ::LogLink, η::Real)
    g  = exp(η)
    return (g, g, sqrt(η))
end
crossprod(obj::AbstractVector{<:Number}) = obj'obj
crossprod(obj::AbstractMatrix{<:Real}) = Hermitian(obj'obj)

function irls(A::AbstractMatrix{<:BlasReal},
              b::AbstractVector{<:BlasReal},
              distribution::Distribution,
              link::AbstractLink)
    F = qrfact(A)
    Q = Matrix(F.Q)
    m, n = size(A)
    x = Vector{Float64}(undef, n)
    η = Vector{Float64}(undef, m)
    w = Vector{Float64}(undef, m)
    for itt ∈ 1:50
        μ, ∂μ∂η, σ = μ∂μ∂ησ.(distribution, link, η)
        z  = η + (b - μ) ./ ∂μ∂η
        w  = ∂μ∂η ./ σ
        x₀ = x
        C  = cholfact!(crossprod(w .* Q)).factors
        x  = LowerTriangular(C') \ (Q' * (w .* z))
        x  = UpperTriangular(C) \ x
        η  = Q * x
        norm(x - x₀, 2) < 1e-8 && return F \ η
    end
end
export
    irls
end
