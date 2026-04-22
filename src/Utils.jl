using AbstractRobustClassifiers
const ARC = AbstractRobustClassifiers

function predict(M::ARC.Maximality, fitresult, x)
	dom = compute_dominance_matrix(M, fitresult, x)
	pred::ARC.Prediction = ARC.Prediction(dom)
	return pred.undominated
end

function compute_dominance_matrix(M::ARC.Maximality, (cond_prob, y_prob, decode_y, _, _, _, _), x)
	dom::ARC.DominanceMatrix = zeros(Bool, size(y_prob,1), size(y_prob,1))
	for y1 in 1:size(y_prob,1), y2 in 1:size(y_prob,1)
		(y1 == y2) && (continue)
		dom[y1, y2] = maximality_dominance(cond_prob, y_prob, x, y1, y2)
	end
	return dom
end

function maximality_dominance(cond_prob, y_prob, x, y1::Int64, y2::Int64)
	log10(y_prob[y1,1]) - log10(y_prob[y2, 2]) + mapreduce(n -> log10(cond_prob[n][x[n],y1,1]) - log10(cond_prob[n][x[n],y2,2]), +, range(1,length(x))) > 0.0
end

function predict(M::ARC.IntervalDominance, fitresult, x)
	@error("Predict method not implemented for decisionRule $M")
end

function compute_dominance_matrix(M::ARC.IntervalDominance, fitresult, x)
	@error("compute_dominance_matrix method not implemented for decisionRule $M")
end

function predict(M::ARC.EAdmissibility, fitresult, x)
	@error("Predict method not implemented for decisionRule $M")
end

function compute_dominance_matrix(M::ARC.EAdmissibility, fitresult, x)
	@error("compute_dominance_matrix method not implemented for decisionRule $M")
end

function predict(M::ARC.GammaMaxiMax, fitresult, x)
	@error("Predict method not implemented for decisionRule $M")
end

function compute_dominance_matrix(M::ARC.GammaMaxiMax, fitresult, x)
	@error("compute_dominance_matrix method not implemented for decisionRule $M")
end

function predict(M::ARC.GammaMaxiMin, fitresult, x)
	@error("Predict method not implemented for decisionRule $M")
end

function compute_dominance_matrix(M::ARC.GammaMaxiMin, fitresult, x)
	@error("compute_dominance_matrix method not implemented for decisionRule $M")
end

function predict(M::ARC.Hurwicz, fitresult, x)
	@error("Predict method not implemented for decisionRule $M")
end

function compute_dominance_matrix(M::ARC.Hurwicz, fitresult, x)
	@error("compute_dominance_matrix method not implemented for decisionRule $M")
end