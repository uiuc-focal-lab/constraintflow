type base_type = Int | Float | Bool | PolyExp | ZonoExp
type non_shape_type = Base of base_type | Array of base_type | Neuron | Nlist | Noise 

type binop = Plus | Minus | Mult | Div | And | Or | Eq | Lt
type unop = Not | Neg


type neuron_metadata = Weight | Bias | Layer 
type nlist_ops = Max | Min | Argmax | Argmin 

type expr = 
| Var of string 
| ConstInt of int
| ConstFloat of float
| ConstBool of bool
| Epsilon
|	Binary of binop * expr * expr 
| Unary of unop * expr 
| Ternary of expr * expr * expr
| GetMetadata of expr * neuron_metadata
| GetElement of expr * string 
| Coeff of expr
| Traverse of expr * string * string * string
| NlistOp of expr * nlist_ops * string
| Sum of expr 
| Sub of expr * expr 
| Map of expr * string 
| Dot of expr * expr 
| FuncCall of string * (expr list) 


type shape_decl = DefShape of (non_shape_type * string) list

type func_decl = DeclFunc of string * (non_shape_type * string) list

type operator = 
| Bot
| Affine 
| Relu
| Maxpool
| Sigmoid
| Tanh

type forward_decl = DeclForward of string * operator * string * string
type forward_ret = Tuple of expr list | If of expr * forward_ret * forward_ret

type statement = 
| Func of func_decl * expr
| Forward of forward_decl * forward_ret
| Seq of statement * statement

type program = Program of shape_decl * statement