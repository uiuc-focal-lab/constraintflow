%{
open Ast 
%}

%token <int> INT
%token <float> FLOAT
%token <string> VAR
%token TRUE FALSE
%token PLUS MINUS MULT DIV AND OR LT EQ EQQ NEQ GT LEQ GEQ NOT 
%token LPAREN RPAREN LSQR RSQR SEMI QUES COLON COMMA DOTT
%token IF TRAV SUM SUB MAP DOT ARGMIN ARGMAX MIN MAX WEIGHT BIAS LAYER 
%token AFFINE RELU MAXPOOL SIGMOID TANH 
%token INTT FLOATT BOOL POLYEXP ZONOEXP NEURON NLIST NOISE
%token SHAPE FUNC FORWARD EPSILON 
// %token PLUS MINUS MULT DIV LT AND OR NOT EQ  EQ NEQQ GT LE GE
// %token LPAREN RPAREN LCURL RCURL LSQR RSQR SEMI IF ELSE WHILE PRE POST INV MALLOC 
%token EOF
%left OR 
%left AND
%left LT GT LEQ GEQ EQ NEQ EQQ
%left PLUS MINUS        /* lowest precedence */
%left MULT        /* medium precedence */
%left DIV        /* medium precedence */
%nonassoc NOT        /* highest precedence */
%start <Ast.program> main             /* the entry point */
%%


main:
    prog EOF                { $1 }
;

prog : 
    | shape_decl statement { Program($1, $2) }
;

shape_decl : 
    | SHAPE LPAREN arglist RPAREN SEMI { DefShape($3) }
;

statement :
    | FUNC func_decl EQ expr SEMI { Func($2, $4) }
    | FORWARD forward_decl EQ forward_ret SEMI { Forward($2, $4) }
    | statement statement { Seq($1, $2) }
;

func_decl :
    | VAR LPAREN arglist RPAREN { DeclFunc($1, $3) }
;

forward_decl :
    | VAR LPAREN operator COMMA NEURON VAR COMMA NLIST VAR RPAREN { DeclForward($1, $3, $6, $9) }
;

operator :
    | AFFINE    {Affine}
    | RELU      {Relu}
    | MAXPOOL   {Maxpool}
    | SIGMOID   {Sigmoid}
    | TANH      {Tanh}
;

forward_ret :
    | IF LPAREN expr COMMA forward_ret COMMA forward_ret { If($3, $5, $7) } 
    | LPAREN expr_list RPAREN { Tuple($2) }
;

expr_list :
    | expr COMMA expr_list { $1 :: $3 }
    | expr { [$1] }
;

expr:
    | FALSE                   { ConstBool(false) }
    | TRUE                    { ConstBool(true) }
    | INT                     { ConstInt($1) }
    | FLOAT                   { ConstFloat($1) }
    | VAR                     { Var($1) }
    | EPSILON                 { Epsilon }
    | LPAREN expr RPAREN      { $2 }
    | expr binop expr         { Binary($2, $1 , $3) }
    | expr LT expr            { Binary(Lt, $1, $3) }
    | expr GT expr            { Binary(Lt, $3, $1) }
    | expr GEQ expr           { Unary(Not, Binary(Lt, $1, $3)) }
    | expr LEQ expr           { Unary(Not, Binary(Lt, $3, $1)) }
    | expr EQQ expr           { Binary(Eq, $1, $3) }
    | expr NEQ expr           { Unary(Not, Binary(Eq, $1, $3)) }
    | NOT expr %prec NOT      { Unary(Not, $2) }
    | expr QUES expr COLON expr {Ternary($1, $3, $5)}
    | expr LSQR metadata RSQR {GetMetadata($1, $3)}
    | expr LSQR VAR RSQR      {GetElement($1, $3)}
    | expr DOT TRAV VAR VAR VAR {Traverse($1, $4, $5, $6)}
    | func_op LPAREN expr COMMA VAR RPAREN { NlistOp($3, $1, $5)}
    | SUM LPAREN expr RPAREN { Sum($3)}
    | SUB LPAREN expr COMMA expr RPAREN { Sub($3, $5)}
    | expr DOT MAP VAR { Map($1, $4)}
    | expr DOT DOTT expr { Dot($1, $4)}
    | VAR LPAREN expr_list RPAREN { FuncCall($1, $3)}
;
  
func_op :
    | MAX       {Max}
    | MIN       {Min}
    | ARGMAX    {Argmax}
    | ARGMIN    {Argmin}
;

binop :
    | PLUS      { Plus }
    | MINUS     { Minus }
    | MULT      { Mult }
    | DIV       { Div }
    | AND       { And }
    | OR        { Or }
;

metadata :
    | WEIGHT    { Weight }
    | BIAS      { Bias }
    | LAYER     { Layer }
;

arglist : 
    | types VAR COMMA arglist { ($1, $2) :: $4 }
    | types VAR {[($1, $2)]}
;

types :
    | INTT      { Base(Int) }
    | FLOATT    { Base(Float) }
    | BOOL      { Base(Bool) }
    | POLYEXP   { Base(PolyExp) }
    | ZONOEXP   { Base(ZonoExp) }
    | NEURON    { Neuron }
    | NLIST     { Nlist }
    | NOISE     { Noise }
;