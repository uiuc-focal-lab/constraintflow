def Shape as (Float l, Float u, PolyExp L, PolyExp U){(curr[l]<=curr) and (curr[L]<=curr) and (curr[u]>=curr) and (curr[U]>=curr)}

func simplify_lower(Neuron n, Float c) = (c>=0) ? (c*n[l]) : (c*n[u])
func simplify_upper(Neuron n, Float c) = (c>=0) ? (c*n[u]) : (c*n[l])
func replace_lower(Neuron n, Float c) = (c>=0) ? (c*n[L]) : (c*n[U])
func replace_upper(Neuron n, Float c) = (c>=0) ? (c*n[U]) : (c*n[L])
func backsubs_lower(PolyExp e, Neuron n) = (e.traverse(backward, priority1, true, replace_lower){e<=n}).map(simplify_lower)
func backsubs_upper(PolyExp e, Neuron n) = (e.traverse(backward, priority1, true, replace_upper){e>=n}).map(simplify_upper)

transformer deeppoly(curr, prev){
    Relu -> sum(prev[l])>=0 ? (sum(prev[l]), sum(prev[u]), sum(prev[L]), sum(prev[U])) : 
    (sum(prev[u])<=0 ? (0,0,0,0) : (0, sum(prev[u]), 0, 
    (sum(prev[u])/(sum(prev[u])-sum(prev[l])))*sum(prev)-(sum(prev[u])*sum(prev[l])/(sum(prev[u])-sum(prev[l])))))

    Affine -> (backsubs_lower(prev.dot(curr[weight]) + curr[bias], curr), backsubs_upper(prev.dot(curr[weight]) + curr[bias], curr), 
    prev.dot(curr[weight]) + curr[bias], prev.dot(curr[weight]) + curr[bias])
}
flow(forward, priority2, true, deeppoly)





