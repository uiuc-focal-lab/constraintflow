grammar dsl;

prog :
    shape_decl statement EOF
;

shape_decl : 
    SHAPE LPAREN arglist RPAREN LBRACE prop RBRACE SEMI 
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

trans_decl : TRANSFORMER VAR LPAREN CURR COMMA PREV RPAREN;     

operator :
    AFFINE    
    | RELU     
    | MAXPOOL   
    | SIGMOID  
    | TANH     
;

trans_ret :
    expr QUES trans_ret COLON trans_ret #condtrans
    | LPAREN trans_ret RPAREN #parentrans
    | expr_list #trans
;

types: INTT 
	|	FLOATT 
	|	BOOL 
	|	POLYEXP 
	|	ZONOEXP 
	|	NEURON 
    | types LIST
    ;

arglist : types VAR COMMA arglist 
	|	types VAR ;

expr_list : expr COMMA expr_list
	|	expr ; 

expr: FALSE					                        #false
    | TRUE 					                        #true
    | IntConst 					                    #int
    | FloatConst 				                    #float
    | VAR                     			            #varExp
    | EPSILON 					                    #epsilon
    | CURR					                        #curr
    | PREV					                        #prev
    | LPAREN expr RPAREN      			            #parenExp
    | expr LSQR metadata RSQR                       #getMetadata
    | expr LSQR VAR RSQR                            #getElement
    | expr binop expr         			            #binopExp
    | NOT expr       				                #not
    | MINUS expr				                    #neg
    | expr QUES expr COLON expr 		            #cond
    | expr DOT TRAV LPAREN direction COMMA expr COMMA expr COMMA expr RPAREN LBRACE prop RBRACE		#traverse
    | argmax_op LPAREN expr COMMA expr RPAREN 	    #argmaxOp
    | max_op LPAREN expr RPAREN                     #maxOpList
    | max_op LPAREN expr COMMA expr RPAREN          #maxOp
    | list_op LPAREN expr RPAREN 			        #listOp
    | expr DOT MAP LPAREN expr RPAREN 		        #map
    | expr DOT DOTT LPAREN expr RPAREN 		        #dot
    | VAR LPAREN expr_list RPAREN 		            #funcCall
;

argmax_op: ARGMAX
	|	ARGMIN ;

max_op: MAX
    |   MIN ;

list_op: SUM
    |   LEN 
    |   AVG ;

binop: PLUS 
	|	MINUS 
	|	MULT 
	|	DIV 
	|	AND 
	|	OR 
	|	GEQ 
	|	LEQ
    |   LT
    |   GT
    |   EQQ 
    ;

metadata: WEIGHT 
	|	BIAS 
	|	LAYER ;

direction: BACKWARD
	|	FORWARD ;

pt: IntConst #ptbasic 
	|	FloatConst #ptbasic
	|	pt LSQR VAR RSQR #ptbasic
	|	pt LSQR metadata RSQR #ptbasic
	|	VAR #ptbasic
	|	CURR #ptbasic
	|	pt PLUS pt #ptop
	|	pt MINUS pt #ptop
	;

prop: LPAREN prop RPAREN #propparen
	|	pt GT pt #propsingle
    |       pt GEQ pt #propsingle
    |       pt LEQ pt #propsingle
    |       pt LT pt #propsingle
    |       pt EQQ pt #propsingle
    |       prop AND prop #propdouble
    |       prop OR prop #propdouble
    |       pt IN pt #ptin
    ;

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
ZONOEXP: 'ZonoExp' ;
NEURON: 'Neuron' ;
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
MAXPOOL: 'Maxpool' ;
SIGMOID: 'Sigmoid' ;
TANH: 'Tanh' ;
SHAPE: 'def Shape as' ;
FUNC: 'func' ;
EPSILON: 'eps' ;
TRUE: 'true' ;
FALSE: 'false' ;
CURR: 'curr' ;
PREV: 'prev' ;

IntConst: Sign? Digit+ ;

FloatConst: [0-9]+'.'[0-9]+([Ee] [+-]? [0-9]+)? ;

fragment Digit : [0-9] ;

fragment Sign : [+-] ; 

VAR : Nondigit (Nondigit | Digit | '\'')* ;
fragment Nondigit : [a-zA-Z_] ;

WS : [ \t\r\n]+ -> skip ;	// skip spaces, tabs, newlines
LineComment : '//' ~[\r\n]* -> channel(HIDDEN) ;
