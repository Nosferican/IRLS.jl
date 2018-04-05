using Documenter, IRLS

makedocs(
    format = :html,
    sitename = "IRLS"
)

deploydocs(
    repo   = "github.com/Nosferican/IRLS.jl.git",
    julia  = "nightly",
    osname = "linux",
    target = "build",
    deps   = nothing,
    make   = nothing
)
