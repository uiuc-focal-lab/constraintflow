grammar dsl;

prog :
    shape_decl statement EOF
;

shape_decl : 
    SHAPE LPAREN arglist RPAREN LBRACE expr RBRACE SEMI 
;

statement : FLOW LPAREN direction COMMA expr COMMA expr COMMA VAR RPAREN SEMI #flowstmt
    | FUNC func_decl EQ expr SEMI #funcstmt
    | transformer #transstmt
    | statement statement #seqstmt
;

func_decl :
    VAR LPAREN arglist RPAREN 
;

transformer: trans_decl LBRACE op_list RBRACE ;

op_list: op_stmt SEMI
	|	op_stmt SEMI op_list
	;

op_stmt: operator ARROW trans_ret ;

trans_decl : TRANSFORMER VAR;     

operator :
    AFFINE    
    | RELU
    | RELU6
    | ABS  
    | HARDTANH   
    | HARDSIGMOID   
    | HARDSWISH   
    | MAXPOOL   
    | MINPOOL   
    | AVGPOOL   
    | NEURON_MULT
    | NEURON_ADD
    | NEURON_MAX
    | NEURON_MIN
    | NEURON_LIST_MULT
    | REVAFFINE   
    | REVRELU      
    | REVRELU6      
    | REVABS      
    | REVHARDSIGMOID      
    | REVHARDTANH      
    | REVHARDSWISH      
    | REVMAXPOOL
    | REVNEURON_MULT
    | REVNEURON_ADD
    | REVNEURON_MAX
    | REVNEURON_MIN     
;

trans_ret :
    expr QUES trans_ret COLON trans_ret #condtrans
    | LPAREN trans_ret RPAREN #parentrans
    | expr_list #trans
;

types:  INTT 
	|	FLOATT 
	|	BOOL 
	|	POLYEXP 
	|	ZONOEXP 
	|	NEURON 
    |   NOISE
	|	CT  
    |   types LIST
    ;

arglist : types VAR COMMA arglist 
	|	types VAR ;

expr_list : expr COMMA expr_list
	|	expr ; 

exprs: expr exprs
    | expr;

expr: FALSE					                        #false
    | TRUE 					                        #true
    | IntConst 					                    #int
    | FloatConst 				                    #float
    | VAR                     			            #varExp
    | EPSILON 					                    #epsilon
    | CURR					                        #curr
    | PREV					                        #prev
    | PREV_0					                    #prev_0
    | PREV_1					                    #prev_1
    | CURRLIST					                    #curr_list
    | LPAREN expr RPAREN      			            #parenExp
    | LSQR expr_list RSQR                           #exprarray
    | expr LSQR metadata RSQR                       #getMetadata
    | expr LSQR VAR RSQR                            #getElement
    | expr binop expr         			            #binopExp
    | NOT expr       				                #not
    | MINUS expr				                    #neg
    | expr QUES expr COLON expr 		            #cond
    | expr DOT TRAV LPAREN direction COMMA expr COMMA expr COMMA expr RPAREN LBRACE expr RBRACE		#traverse
    | argmax_op LPAREN expr COMMA expr RPAREN 	    #argmaxOp
    | max_op LPAREN expr RPAREN                     #maxOpList
    | max_op LPAREN expr COMMA expr RPAREN          #maxOp
    | list_op LPAREN expr RPAREN 			        #listOp
    | expr DOT MAP LPAREN expr RPAREN 		        #map
    | expr DOT MAPLIST LPAREN expr RPAREN 		    #map_list
    | expr DOT DOTT LPAREN expr RPAREN 		        #dot
    | expr DOT CONCAT LPAREN expr RPAREN 		    #concat
    | LP LPAREN lp_op COMMA expr COMMA expr RPAREN  #lp
    | VAR LPAREN expr_list RPAREN 		            #funcCall
    | VAR exprs                                     #curry
;

argmax_op: ARGMAX
	|	ARGMIN ;

lp_op:  'minimize'
	|	'maximize' ;

max_op: MAX
    |   MIN ;

list_op: SUM
    |   LEN 
    |   AVG ;

binop: DIV 
	|	MULT  
	|	MINUS 
	|	PLUS  
	|	AND 
	|	OR 
	|	GEQ 
	|	LEQ
    |   LT
    |   GT
    |   EQQ 
    |   IN
    ;

metadata: WEIGHT 
	|	BIAS 
	|	EQUATIONS 
	|	LAYER ;

direction: BACKWARD
	|	FORWARD ;

FLOW: 'flow' ;
ARROW: '->' ;
TRANSFORMER: 'transformer' ;
IN: 'In' ;
OUT: 'out' ;
BACKWARD: 'backward' ;
FORWARD: 'forward' ;
INTT: 'Int' ;
FLOATT: 'Float' ;
BOOL: 'Bool' ;
POLYEXP: 'PolyExp' ;
ZONOEXP: 'SymExp' ;
NEURON: 'Neuron' ;
NOISE: 'Noise' ;
CT: 'Ct' ;
LIST: 'List' ;
DOT: '.' ;
COMMA: ',' ;
PLUS: '+' ;
MINUS: '-' ;
MULT: '*' ;
DIV: '/' ;
AND: 'and' ;
OR: 'or' ;
LT: '<' ;
EQ: '=' ;
EQQ: '==' ;
NEQ: '!=' ;
GT: '>' ;
LEQ: '<=' ;
GEQ: '>=' ;
NOT: '!' ;
LPAREN: '(' ;
RPAREN: ')' ;
LSQR: '[' ;
RSQR: ']' ;
LBRACE: '{' ;
RBRACE: '}' ;
SEMI: ';' ;
QUES: '?' ;
COLON: ':' ;
IF: 'if' ;
TRAV: 'traverse' ;
SUM: 'sum' ;
LEN: 'len' ;
AVG: 'avg' ;
SUB: 'sub' ;
MAP: 'map' ;
MAPLIST: 'map_list' ;
DOTT: 'dot' ;
ARGMIN: 'argmin' ;
ARGMAX: 'argmax' ;
MIN: 'min' ;
MAX: 'max' ;
WEIGHT: 'weight' ;
BIAS: 'bias' ;
LAYER: 'layer' ;
AFFINE: 'Affine' ;
RELU: 'Relu' ;
RELU6: 'Relu6' ;
ABS: 'Abs' ;
HARDTANH: 'HardTanh' ;
HARDSIGMOID: 'HardSigmoid' ;
HARDSWISH: 'HardSwish' ;
MAXPOOL: 'Maxpool' ;
MINPOOL: 'Minpool' ;
AVGPOOL: 'Avgpool' ;
REVAFFINE: 'rev_Affine' ;
NEURON_MULT: 'Neuron_mult' ;
NEURON_ADD: 'Neuron_add' ;
NEURON_MAX: 'Neuron_max' ;
NEURON_MIN: 'Neuron_min' ;
NEURON_LIST_MULT: 'Neuron_list_mult' ;
REVRELU: 'rev_Relu' ;
REVRELU6: 'rev_Relu6' ;
REVABS: 'rev_Abs' ;
REVHARDSIGMOID: 'rev_HardSigmoid' ;
REVHARDTANH: 'rev_HardTanh' ;
REVHARDSWISH: 'rev_HardSwish' ;
REVMAXPOOL: 'rev_Maxpool' ;
REVNEURON_MULT: 'rev_Neuron_mult' ;
REVNEURON_ADD: 'rev_Neuron_add' ;
REVNEURON_MAX: 'rev_Neuron_max' ;
REVNEURON_MIN: 'rev_Neuron_min' ;
SIGMOID: 'Sigmoid' ;
TANH: 'Tanh' ;
SHAPE: 'def Shape as' ;
FUNC: 'func' ;
EPSILON: 'eps' ;
TRUE: 'true' ;
FALSE: 'false' ;
CURR: 'curr' ;
PREV: 'prev' ;
PREV_0: 'prev_0' ;
PREV_1: 'prev_1' ;
CURRLIST: 'curr_list' ;
LP: 'lp' ;
CONCAT: 'concat' ;
EQUATIONS: 'equations' ;

IntConst: Sign? Digit+ ;

FloatConst: [0-9]+'.'[0-9]+([Ee] [+-]? [0-9]+)? ;

fragment Digit : [0-9] ;

fragment Sign : [+-] ; 

VAR : Nondigit (Nondigit | Digit | '\'')* ;
fragment Nondigit : [a-zA-Z_] ;

WS : [ \t\r\n]+ -> skip ;	// skip spaces, tabs, newlines
LineComment : '//' ~[\r\n]* -> channel(HIDDEN) ;
