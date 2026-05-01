using RobustClassifiersBase
const RCB = RobustClassifiersBase

function RCB.predict(M::RCB.Maximality, fitresult, x)
	dom = RCB.compute_dominance_matrix(M, fitresult, x)
	pred::RCB.Prediction = RCB.Prediction(dom)
	return pred.undominated
end

function RCB.compute_dominance_matrix(M::RCB.Maximality, (cond_prob, y_prob, decode_y, _, _, _, _), x)
	dom::RCB.DominanceMatrix = zeros(Bool, size(y_prob,1), size(y_prob,1))
	for y1 in 1:size(y_prob,1), y2 in 1:size(y_prob,1)
		(y1 == y2) && (continue)
		dom[y1, y2] = maximality_dominance(cond_prob, y_prob, x, y1, y2)
	end
	return dom
end

function maximality_dominance(cond_prob, y_prob, x, y1::Int64, y2::Int64)
	log10(y_prob[y1,1]) - log10(y_prob[y2, 2]) + mapreduce(n -> log10(cond_prob[n][x[n],y1,1]) - log10(cond_prob[n][x[n],y2,2]), +, range(1,length(x))) > 0.0
end

function RCB.predict(M::RCB.IntervalDominance, fitresult, x)
	@error("Predict method not implemented for decisionRule $M")
end

function RCB.compute_dominance_matrix(M::RCB.IntervalDominance, fitresult, x)
	@error("compute_dominance_matrix method not implemented for decisionRule $M")
end

function RCB.predict(M::RCB.EAdmissibility, fitresult, x)
	@error("Predict method not implemented for decisionRule $M")
end

function RCB.compute_dominance_matrix(M::RCB.EAdmissibility, fitresult, x)
	@error("compute_dominance_matrix method not implemented for decisionRule $M")
end

function RCB.predict(M::RCB.GammaMaxiMax, fitresult, x)
	@error("Predict method not implemented for decisionRule $M")
end

function RCB.compute_dominance_matrix(M::RCB.GammaMaxiMax, fitresult, x)
	@error("compute_dominance_matrix method not implemented for decisionRule $M")
end

function RCB.predict(M::RCB.GammaMaxiMin, fitresult, x)
	@error("Predict method not implemented for decisionRule $M")
end

function RCB.compute_dominance_matrix(M::RCB.GammaMaxiMin, fitresult, x)
	@error("compute_dominance_matrix method not implemented for decisionRule $M")
end

function RCB.predict(M::RCB.Hurwicz, fitresult, x)
	@error("Predict method not implemented for decisionRule $M")
end

function RCB.compute_dominance_matrix(M::RCB.Hurwicz, fitresult, x)
	@error("compute_dominance_matrix method not implemented for decisionRule $M")
end