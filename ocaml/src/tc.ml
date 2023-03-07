open Ast 

exception Not_joinable of (operator * operator)
exception Incompatible_shape
exception Not_implemented
exception Unknown_var of string 
exception Type_mismatch
exception Multiplication_not_allowed
exception Duplicate_variable of string 

type types = Non_product of non_shape_type | Product of (non_shape_type list) * non_shape_type | Shape of ((non_shape_type * int) list)

let joinable x y = 
  match x, y with 
  | Bot, _ -> true
  | _, Bot -> true
  | _, _ -> if x=y then true else false 

let join x y = 
  match x, y with 
  | Bot, _ -> y 
  | _, Bot -> x 
  | _, _ -> if (joinable x y) then x else raise (Not_joinable(x, y))

let join_list l = List.fold_left join Bot l  

let subtype_base x y = 
  match x, y with 
  | Int, Float -> true 
  | Int, PolyExp -> true 
  | Int, ZonoExp -> true 
  | Float, PolyExp -> true 
  | Float, ZonoExp -> true
  | _, _ -> false 

let subtype x y = 
  match x, y with 
  | Neuron, Base(PolyExp) -> true
  | Noise, Base(ZonoExp) -> true
  | Base(x1), Base(y1) -> subtype_base x1 y1
  | Array(x1), Array(y1) -> subtype_base x1 y1 
  | _, _ -> false

let rec matching_types l1 l2 = 
  match l1, l2 with 
  | [], [] -> true
  | x::ls1, y::ls2 -> if (subtype x y) || (subtype y x) then matching_types ls1 ls2 else false 
  | _, _ -> raise Incompatible_shape

let rec typeof env tau_s l_par expr = 
  match expr with
  | ConstBool(_) -> Base Bool, Bot, 0 
  | ConstInt(_) -> Base Int, Bot, 0 
  | ConstFloat(_) -> Base Float, Bot, 0
  | Var x -> typeof_var env x
  | Epsilon -> Noise, Bot, 1
  | Binary(b, e1, e2) -> typeof_binop env tau_s l_par b e1 e2
  | Unary(_, e1) -> typeof_unary env tau_s l_par e1
  | Ternary(e1, e2, e3) -> typeof_ternary env tau_s l_par e1 e2 e3
  | GetMetadata(e1, m) -> typeof_metadata env tau_s l_par e1 m 
  | GetElement(e1, x) -> typeof_element env tau_s l_par e1 x
  | Traverse(e1, x1, x2, x3) -> typeof_traverse env tau_s l_par e1 x1 x2 x3 
  | NlistOp(e1, op, x) -> typeof_nlist_op env tau_s l_par e1 op x
  | Sum(e1) -> typeof_sum env tau_s l_par e1 
  | Sub(e1, e2) -> typeof_sub env tau_s l_par e1 e2
  | Map(e1, x) -> typeof_map env tau_s l_par e1 x 
  | Dot(e1, e2) -> typeof_dot env tau_s l_par e1 e2 
  | FuncCall(x, elist) -> typeof_func_call env tau_s l_par x elist 
  | _ -> raise Not_implemented

and typeof_func_call env tau_s l_par x elist = 
  let tau, l, len = Hashtbl.find env x in 
  let l_types = List.map (typeof env tau_s l_par) elist in 
  let l_layers = List.map (fun (_,y,_) -> y) l_types in 
  let l_types = List.map (fun (x,_,_) -> x) l_types in 
  let l = join_list (l::l_layers) in 
  match tau with 
  | Product(tau_list, t) -> let _ = (matching_types tau_list l_types) in t, l, len 
  | _ -> raise Type_mismatch

and typeof_dot env tau_s l_par e1 e2 = 
  let tau1, l1, len1 = typeof env tau_s l_par e1 in 
  let tau2, l2, len2 = typeof env tau_s l_par e2 in 
  let l = join_list [l1; l2] in
  if (((subtype tau2 (Array Float)) && (len1=len2)) || ((subtype tau2 (Base Float)) && (len2=0))) && (tau1=Nlist) then 
    Base PolyExp, l, len1 
  else
    raise Type_mismatch

and typeof_map env tau_s l_par e1 x = 
  let tau1, l1, _ = typeof env tau_s l_par e1 in 
  let tau2, l2, len2 = Hashtbl.find env x in 
  let l = join_list [l1; l2] in 
  match tau1, tau2 with 
  | Base PolyExp, Product([Neuron ; Base Float], t) -> t, l, len2
  | Base ZonoExp, Product([Noise ; Base Float], t) -> t, l, len2
  | Array t1, Product([Base t2], t) when t1=t2 -> t, l, len2
  | _ , _ -> raise Type_mismatch

and typeof_sub env tau_s l_par e1 e2 = 
  let tau1, l1, len1 = typeof env tau_s l_par e1 in 
  let tau2, l2, len2 = typeof env tau_s l_par e2 in
  let l = join_list [l1; l2] in 
  match tau1, tau2 with 
  | Nlist, Neuron -> tau1, l, len1-len2
  | _ , _ -> raise Type_mismatch

and typeof_sum env tau_s l_par e1 = 
  let tau, l, len = typeof env tau_s l_par e1 in 
    match tau with 
      | Nlist -> Base PolyExp, l, len
      | Array Bool -> raise Type_mismatch
      | Array t -> Base t, l, 0 
      | _ -> raise Type_mismatch 

and typeof_nlist_op env tau_s l_par e1 op x = 
  let tau1, l, _ = typeof env tau_s l_par e1 in 
  let tau2, len2 = Hashtbl.find tau_s x in 
  match op, tau1 with 
    | Min, Nlist -> tau2, l, len2
    | Max, Nlist -> tau2, l, len2
    | Argmin, Nlist -> Neuron, l, 1
    | Argmax, Nlist -> Neuron, l, 1
    | _ , _ -> raise Type_mismatch

and typeof_traverse env tau_s l_par e x1 x2 x3 = 
  let tau, l, len = typeof env tau_s l_par e in
  let tau1, l1, _ = Hashtbl.find env x1 in 
  let tau2, l2, _ = Hashtbl.find env x2 in 
  let tau3, l3, _ = Hashtbl.find env x3 in
  let l = join_list [l; l1; l2; l3] in 
  match tau, tau1, tau2, tau3 with 
    | Base PolyExp , Product([Neuron ; Base Float], Base Bool), Product([Neuron ; Base Float], Base Float), Product([Neuron ; Base Float], tau4) -> if (subtype tau4 (Base PolyExp)) then (Base PolyExp), l, len else raise Type_mismatch
    | _ , _ , _ , _ -> raise Type_mismatch

and typeof_element env tau_s l_par e1 x =
  let tau1, l1, len1 = typeof  env tau_s l_par e1 in 
  let tau2, len2 = Hashtbl.find tau_s x in 
  match tau1 with 
  | Neuron -> tau2, l1, len2 
  | Nlist -> (match tau2 with 
    | Base t -> Array t, l1, len1
    | _ -> raise Not_implemented
  )
  | _ -> raise Type_mismatch

and typeof_metadata env tau_s l_par e1 m = 
  let tau1, l1, len1 = typeof  env tau_s l_par e1 in 
  match tau1 with 
  | Neuron -> (match m with 
    | Weight -> (match l1 with 
      | Affine -> Array Float, Affine, l_par 
      | _ -> raise Type_mismatch)
    | Bias -> (match l1 with 
      | Affine -> Base Float, Affine, 0 
      | _ -> raise Type_mismatch)
    | Layer -> Base(Int), l1, 0)
  | Nlist -> (match m with 
    | Weight -> (match l1 with 
      | Affine -> Array Float, Affine, l_par*len1 
      | _ -> raise Type_mismatch)
    | Bias -> (match l1 with 
      | Affine -> Base Float, Affine, len1  
      | _ -> raise Type_mismatch)
    | Layer -> Base(Int), l1, len1)
  | _ -> raise Type_mismatch

and typeof_ternary env tau_s l_par e1 e2 e3 = 
  let tau1, l1, _ = typeof env tau_s l_par e1 in 
  let tau2, l2, len2 = typeof env tau_s l_par e2 in 
  let tau3, l3, len3 = typeof env tau_s l_par e3 in 
  let l = join_list [l1 ; l2 ; l3] in
  if tau1=Base(Bool) then 
    if subtype tau2 tau3 then 
      tau3, l, (max len2 len3)
    else
      if subtype tau3 tau2 then 
        tau2, l, (max len2 len3)
      else
        raise Type_mismatch  
  else 
    raise Type_mismatch

and typeof_unary env tau_s l_par e1 = 
  let tau1, l1, len1 = typeof env tau_s l_par e1 in 
  if tau1=Base(Bool) then 
    tau1, l1, len1
  else 
    raise Type_mismatch    

and typeof_binop env tau_s l_par b e1 e2 = 
  let tau1, l1, len1 = typeof env tau_s l_par e1 in 
  let tau2, l2, len2 = typeof env tau_s l_par e2 in 
  let l = join_list [l1 ; l2] in
  match b with 
  | Eq | Lt -> typeof_comparison tau1 tau2 l
  | Mult | Div -> typeof_mult tau1 tau2 l len1 len2  
  | Plus | Minus -> typeof_plus tau1 tau2 l len1 len2
  | And | Or -> typeof_and tau1 tau2 l 

and typeof_and tau1 tau2 l = if tau1=Base(Bool) && (tau2=Base(Bool)) then tau1, l, 0 else raise Type_mismatch

and typeof_plus tau1 tau2 l len1 len2 = 
  if subtype tau1 tau2 then 
    tau2, l, len1+len2 
  else 
    if subtype tau2 tau1 then 
      tau1, l, len1+len2 
    else 
      raise Type_mismatch

and typeof_mult tau1 tau2 l len1 len2 = 
  if tau1 = tau2 then 
    match tau1 with 
    | Base(PolyExp) -> raise Multiplication_not_allowed
    | Base(ZonoExp) -> raise Multiplication_not_allowed
    | _ -> tau1, l, (max len1 len2)
  else 
    if subtype tau1 tau2 then 
        tau2, l, len2 
    else 
      if subtype tau2 tau1 then 
        tau1, l, len1 
      else 
        raise Type_mismatch

and typeof_comparison tau1 tau2 l = 
  if (subtype tau1 tau2) || (subtype tau2 tau1) then Base(Bool), l, 0 else raise Type_mismatch

and typeof_var env x = try 
  (match (Hashtbl.find env x) with 
  | Non_product(t), l, len -> t, l, len 
  | _ -> raise Type_mismatch)
  with Not_found -> raise (Unknown_var(x)) 


let rec typecheck env tau_s l_par l_ct s = 
  match s with 
  | Func(func_decl, e) -> typecheck_func env tau_s l_par l_ct func_decl e 
  | Forward(forward_decl, forward_ret) -> typecheck_forward env tau_s l_par forward_decl forward_ret
  | Seq(s1, s2) -> typecheck_seq env tau_s l_par l_ct s1 s2 

and typecheck_seq env tau_s l_par l_ct s1 s2  = 
  let _ = typecheck env tau_s l_par l_ct s1 in 
  typecheck env tau_s l_par l_ct s2

and typecheck_forward env tau_s l_par forward_decl forward_ret = 
  let env', name, op = typecheck_forward_decl env l_par forward_decl in 
  if Hashtbl.mem env name then 
    raise (Duplicate_variable name)  
  else 
    let t = typecheck_forward_ret env' tau_s op l_par forward_ret in
    Hashtbl.add env name (Shape t, op, 0)

and typecheck_forward_decl env l_par forward_decl = 
  let env' = Hashtbl.copy env in 
  match forward_decl with 
  | DeclForward(name, op, x2, x3) -> (
    let _ = Hashtbl.add env' x2 ((Non_product Neuron), Bot, 1) in 
    let _ = Hashtbl.add env' x3 ((Non_product Nlist), Bot, l_par) in 
    env', name, op
  )

and typecheck_forward_ret env tau_s op l_par forward_ret = 
  match forward_ret with 
  | Tuple(elist) -> (
    let t = List.map (typeof env tau_s l_par) elist in 
    let _ = join_list (op :: (List.map (fun (_, y, _) -> y) t)) in 
    let e_types = List.map (fun (x, _, _) -> x) t in 
    let t2 = Hashtbl.fold (fun _ (y, _) z -> y::z) tau_s [] in 
    if not(List.fold_left (fun x y -> x && y) true (List.map2 (fun x y -> (subtype x y)) e_types t2)) then 
      raise Type_mismatch
    else
      List.map (fun (x, _, z) -> (x, z)) t
  )
  | If(e, f_ret1, f_ret2) -> (
    let tau1, l1, _ = typeof env tau_s l_par e in 
    if not(tau1=Base(Bool)) then 
      raise Type_mismatch
    else
      let _ = join_list [l1; op] in 
      let t1 = typecheck_forward_ret env tau_s op l_par f_ret1 in  
      let t2 = typecheck_forward_ret env tau_s op l_par f_ret2 in  
      List.map2 (fun (x, y) (_, w) -> (x, (max y w))) t1 t2
  )

and typecheck_func env tau_s l_par l_ct func_decl e = 
  let env' = Hashtbl.copy env in 
  match func_decl with 
    | DeclFunc(name, arglist) -> let _ = (List.map (fun (x, y) -> (Hashtbl.add env' y (get_t x l_par l_ct))) arglist ) in 
      let tau, l, len = typeof env' tau_s l_par e in 
      let e_types = List.map (fun (x, _) -> x) arglist in 
      if Hashtbl.mem env name then 
        raise (Duplicate_variable name)
      else
        Hashtbl.add env name (Product(e_types, tau), l, len)

and get_t x l_par l_ct = match x with 
  | Base(PolyExp) -> Non_product x, Bot, l_ct 
  | Base(ZonoExp) -> Non_product x, Bot, l_ct 
  | Nlist -> Non_product x, Bot, l_par 
  | _ -> Non_product x, Bot, 0

let rec typecheck_program l_par l_ct p = 
  match p with 
  | Program(k, s) -> (
    let tau_s = typecheck_shape l_par l_ct k in 
    let env = Hashtbl.create 1234 in 
    typecheck env tau_s l_par l_ct s 
  ) 

and typecheck_shape l_par l_ct k = 
  let tau_s = Hashtbl.create 10 in 
  match k with 
  | DefShape(arglist) -> let _ = List.map (fun (x, y) -> if Hashtbl.mem tau_s y then raise (Duplicate_variable y) else Hashtbl.add tau_s y (get_t2 x l_par l_ct)) arglist in 
  tau_s

and get_t2 x l_par l_ct = match x with 
  | Base(PolyExp) -> x, l_ct 
  | Base(ZonoExp) -> x, l_ct 
  | Nlist -> x, l_par 
  | _ -> x, 0