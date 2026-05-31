using CategoricalArrays
using MLJModelInterface
const MMI = MLJModelInterface

MMI.@mlj_model mutable struct NCClassifier <: MMI.Deterministic
	s::Float64 = 2.0::(_ > 0)                                   # imprecise parameter
	epsilon::Float64 = 0.05::(_ > 0 && _ < 1)                   # mixture factor in ϵ-contaminated to avoid 0 probabilities
    decisionRule::RCB.DecisionRuleTypes = RCB.Maximality() # 
end

function MMI.fit(m::NCClassifier, verbosity::Int, X, y)
    decode_y = MMI.decoder(y[1])
    decode_x = tuple(collect(MMI.decoder(x) for x in X[1,:])...)
    names_x = names(X[1,:])

    n_y = length(levels(y))
    n_x = [length(levels(X[:,x])) for x in range(1,size(X,2))]
    y_l = MMI.int(y)
    x_l = MMI.int.(X)
    cond_count = [zeros(Int64, (n_x[x], n_y)) for x in range(1,size(X,2))]
    y_count = zeros(Int64, n_y)
    for row in range(1,size(X,1))
        y_count[y_l[row]] += 1
        for col in range(1,size(X,2))
            cond_count[col][x_l[row,col],y_l[row]] += 1
        end
    end

    cond_prob = [zeros(Float64, (n_x[x], n_y, 2)) for x in range(1,size(X,2))]
    y_prob = zeros(Float64, (n_y,2))

    for col in range(1,size(X,2)), val in range(1, n_x[col]), y_cond in range(1,n_y)
        cond_prob[col][val, y_cond, 1] = (1-m.epsilon) * cond_count[col][val,y_cond] * (1/(sum(cond_count[col][:,y_cond])+m.s)) + m.epsilon / n_x[col]
        cond_prob[col][val, y_cond, 2] = (1-m.epsilon) * (cond_count[col][val,y_cond] + m.s) * (1/(sum(cond_count[col][:,y_cond])+m.s)) + m.epsilon / n_x[col]
    end
    for val in range(1,n_y)
        y_prob[val,1] = (1 - m.epsilon) * y_count[val] * (1 / (sum(y_count) + m.s)) .+ m.epsilon / n_y
        y_prob[val,2] = (1 - m.epsilon) * (y_count[val] + m.s) * (1 / (sum(y_count) + m.s)) .+ m.epsilon / n_y
    end

    cache = nothing
    report = nothing
    return (cond_prob, y_prob, decode_y, decode_x, names_x, cond_count, y_count), cache, report
end

function MMI.predict(m::NCClassifier, fitresult, Xnew)
    (_, _, decode_y, _, _, _, _) = fitresult
	y_hat = Vector{CategoricalValue}[]
	for x in eachrow(Xnew)
		push!(y_hat, decode_y.(RCB.predict(m.decisionRule, fitresult, [Int64(MMI.int(x[i])) for i in range(1, size(x, 1))])))
	end
	return y_hat
end

function MMI.fitted_params(m::NCClassifier, fitresult)
    cond_prob, y_prob, decode_y, decode_x, names_x, cond_count, y_count = fitresult
    return (; x_probas=(;zip(Symbol.(names_x), [(; zip(Symbol.(decode_x[x].classes), [Dict("$x_val|$y_val" => [cond_prob[x][x_v, y_v,1], cond_prob[x][x_v,y_v,2]] for (y_v,y_val) in enumerate(decode_y.classes)) for (x_v, x_val) in enumerate(decode_x[x].classes)])...) for x in 1:length(names_x)])...), 
        x_counts=(;zip(Symbol.(names_x), [(; zip(Symbol.(decode_x[x].classes), [Dict("$x_val|$y_val" => cond_count[x][x_v, y_v] for (y_v,y_val) in enumerate(decode_y.classes)) for (x_v, x_val) in enumerate(decode_x[x].classes)])...) for x in 1:length(names_x)])...), 
        y=(;zip(Symbol.(decode_y.classes), [[y_prob[y,1], y_prob[y,2]] for y in 1:size(y_prob,1)])...), 
        y_counts=(;zip(Symbol.(decode_y.classes), [y_count[y] for y in 1:size(y_prob,1)])...),
    )
end


MMI.metadata_pkg(
    NCClassifier,
    name="NaiveCredalClassifier",
    package_uuid="f48f88b4-357d-4ca3-8c39-cae33f76017d",
    package_url="https://github.com/BOB-Henoik/NaiveCredalClassifier.jl",
    package_license="MIT",
    is_pure_julia=true
)

MMI.metadata_model(
    NCClassifier,
    input_scitype=MMI.Table(MMI.OrderedFactor, MMI.Multiclass),
    target_scitype=AbstractVector{<:MMI.Finite},
    output_scitype=AbstractVector{Tuple{<:MMI.Finite}},
    load_path="NaiveCredalClassifier.NCClassifier"
)