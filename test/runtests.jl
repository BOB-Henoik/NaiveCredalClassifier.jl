using NaiveCredalClassifier
using MLJBase
using Test
using CSV
using DataFrames

T = CSV.read("./test/animal_dataset.csv", DataFrame)
T = coerce(T, :hair => Multiclass, :tail => Multiclass, :ear => Multiclass, :animal => Multiclass)
y, X = unpack(T, ==(:animal))

Xnew = DataFrame(ear=["Long"], tail=["Short"], hair=["Long"])
Xnew = coerce(Xnew, :hair => Multiclass, :tail => Multiclass, :ear => Multiclass)
levels!(Xnew.ear, ["Long", "Medium", "Short"])
levels!(Xnew.hair, ["Long", "Medium", "Short"])
levels!(Xnew.tail, ["Long", "Medium", "Short"])
#Xnew =  Xnew[1,:]

ncc = NaiveCredalClassifier.NCClassifier()
mach = machine(ncc, X, y)
fit!(mach)
println(predict(mach, Xnew))
println(fitted_params(mach))
