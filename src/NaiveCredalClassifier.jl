module NaiveCredalClassifier

export 
    NCClassifier,
    fit,
    predict,
    fitted_params,
    predict,
    compute_dominance_matrix
    
include("Utils.jl")
include("NCC.jl")

end
