{
    open Parser        (* The type token is defined in parser.mli *)
    exception Eof
}
rule token = 
    parse
    [' ' '\t' '\n' '\r']     { token lexbuf }     (* skip blanks *)
    |  "//"[' '-'~']*'\n'     {token lexbuf} (* skip comments *)
    | '.'            { DOT }
    | ','            { COMMA }
    | '+'            { PLUS }
    | '-'            { MINUS }
    | '*'            { MULT }
    | '/'            { DIV }
    | '&'            { AND }
    | '|'            { OR }
    | '<'            { LT }
    | "="            { EQ }
    | "=="           { EQQ }
    | "!="           { NEQ }
    | '>'            { GT }
    | "<="           { LEQ }
    | ">="           { GEQ }
    | '!'            { NOT }
    | '('            { LPAREN }
    | ')'            { RPAREN }
    | '['            { LSQR }
    | ']'            { RSQR }
    | ';'            { SEMI }
    | '?'            { QUES }
    | ':'            { COLON }
    | "if"           { IF }
    | "traverse"     { TRAV }
    | "sum"          { SUM }
    | "sub"          { SUB }
    | "map"          { MAP }
    | "dot"          { DOTT }
    | "argmin"       { ARGMIN }
    | "argmax"       { ARGMAX }
    | "min"          { MIN }
    | "max"          { MAX }
    | "weight"       { WEIGHT }
    | "bias"         { BIAS }
    | "layer"        { LAYER }
    | "Affine"       { AFFINE }
    | "Relu"         { RELU }
    | "Maxpool"      { MAXPOOL }
    | "Sigmoid"      { SIGMOID }
    | "Tanh"         { TANH }
    | "int"          { INTT }
    | "float"        { FLOATT }
    | "bool"         { BOOL }
    | "polyexp"      { POLYEXP }
    | "zonoexp"      { ZONOEXP }
    | "neuron"       { NEURON }
    | "nlist"        { NLIST }
    | "noise"        { NOISE }
    | "def Shape as" { SHAPE }
    | "func"         { FUNC }
    | "forward"      { FORWARD }
    | "eps"          { EPSILON }
    | "true"         { TRUE }
    | "false"        { FALSE }
    | ['0'-'9']+ as lxm { INT(int_of_string lxm) }
    | ['0'-'9']+'.'['0'-'9']+ (['E' 'e'] ['+' '-']? ['0'-'9']+)? as lxm { FLOAT(float_of_string lxm) }
    | ['_' 'a'-'z' 'A'-'Z']['_' 'a'-'z' 'A'-'Z' '0'-'9']* as lxm { VAR(lxm) }
    | eof            { EOF }
