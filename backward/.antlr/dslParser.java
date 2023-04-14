// Generated from /home/avaljot/cf2/constraintflow/python_r/dsl.g4 by ANTLR 4.9.2
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class dslParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.9.2", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		FLOW=1, ARROW=2, TRANSFORMER=3, IN=4, OUT=5, BACKWARD=6, FORWARD=7, INTT=8, 
		FLOATT=9, BOOL=10, POLYEXP=11, ZONOEXP=12, NEURON=13, LIST=14, DOT=15, 
		COMMA=16, PLUS=17, MINUS=18, MULT=19, DIV=20, AND=21, OR=22, LT=23, EQ=24, 
		EQQ=25, NEQ=26, GT=27, LEQ=28, GEQ=29, NOT=30, LPAREN=31, RPAREN=32, LSQR=33, 
		RSQR=34, LBRACE=35, RBRACE=36, SEMI=37, QUES=38, COLON=39, IF=40, TRAV=41, 
		SUM=42, LEN=43, AVG=44, SUB=45, MAP=46, DOTT=47, ARGMIN=48, ARGMAX=49, 
		MIN=50, MAX=51, WEIGHT=52, BIAS=53, LAYER=54, AFFINE=55, RELU=56, MAXPOOL=57, 
		SIGMOID=58, TANH=59, SHAPE=60, FUNC=61, EPSILON=62, TRUE=63, FALSE=64, 
		CURR=65, PREV=66, IntConst=67, FloatConst=68, VAR=69, WS=70, LineComment=71;
	public static final int
		RULE_prog = 0, RULE_shape_decl = 1, RULE_statement = 2, RULE_func_decl = 3, 
		RULE_transformer = 4, RULE_op_list = 5, RULE_op_stmt = 6, RULE_trans_decl = 7, 
		RULE_operator = 8, RULE_trans_ret = 9, RULE_types = 10, RULE_arglist = 11, 
		RULE_expr_list = 12, RULE_expr = 13, RULE_argmax_op = 14, RULE_max_op = 15, 
		RULE_list_op = 16, RULE_binop = 17, RULE_metadata = 18, RULE_direction = 19, 
		RULE_pt = 20, RULE_prop = 21;
	private static String[] makeRuleNames() {
		return new String[] {
			"prog", "shape_decl", "statement", "func_decl", "transformer", "op_list", 
			"op_stmt", "trans_decl", "operator", "trans_ret", "types", "arglist", 
			"expr_list", "expr", "argmax_op", "max_op", "list_op", "binop", "metadata", 
			"direction", "pt", "prop"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'flow'", "'->'", "'transformer'", "'In'", "'out'", "'backward'", 
			"'forward'", "'Int'", "'Float'", "'Bool'", "'PolyExp'", "'ZonoExp'", 
			"'Neuron'", "'List'", "'.'", "','", "'+'", "'-'", "'*'", "'/'", "'and'", 
			"'or'", "'<'", "'='", "'=='", "'!='", "'>'", "'<='", "'>='", "'!'", "'('", 
			"')'", "'['", "']'", "'{'", "'}'", "';'", "'?'", "':'", "'if'", "'traverse'", 
			"'sum'", "'len'", "'avg'", "'sub'", "'map'", "'dot'", "'argmin'", "'argmax'", 
			"'min'", "'max'", "'weight'", "'bias'", "'layer'", "'Affine'", "'Relu'", 
			"'Maxpool'", "'Sigmoid'", "'Tanh'", "'def Shape as'", "'func'", "'eps'", 
			"'true'", "'false'", "'curr'", "'prev'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, "FLOW", "ARROW", "TRANSFORMER", "IN", "OUT", "BACKWARD", "FORWARD", 
			"INTT", "FLOATT", "BOOL", "POLYEXP", "ZONOEXP", "NEURON", "LIST", "DOT", 
			"COMMA", "PLUS", "MINUS", "MULT", "DIV", "AND", "OR", "LT", "EQ", "EQQ", 
			"NEQ", "GT", "LEQ", "GEQ", "NOT", "LPAREN", "RPAREN", "LSQR", "RSQR", 
			"LBRACE", "RBRACE", "SEMI", "QUES", "COLON", "IF", "TRAV", "SUM", "LEN", 
			"AVG", "SUB", "MAP", "DOTT", "ARGMIN", "ARGMAX", "MIN", "MAX", "WEIGHT", 
			"BIAS", "LAYER", "AFFINE", "RELU", "MAXPOOL", "SIGMOID", "TANH", "SHAPE", 
			"FUNC", "EPSILON", "TRUE", "FALSE", "CURR", "PREV", "IntConst", "FloatConst", 
			"VAR", "WS", "LineComment"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "dsl.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public dslParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	public static class ProgContext extends ParserRuleContext {
		public Shape_declContext shape_decl() {
			return getRuleContext(Shape_declContext.class,0);
		}
		public StatementContext statement() {
			return getRuleContext(StatementContext.class,0);
		}
		public TerminalNode EOF() { return getToken(dslParser.EOF, 0); }
		public ProgContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_prog; }
	}

	public final ProgContext prog() throws RecognitionException {
		ProgContext _localctx = new ProgContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_prog);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(44);
			shape_decl();
			setState(45);
			statement(0);
			setState(46);
			match(EOF);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Shape_declContext extends ParserRuleContext {
		public TerminalNode SHAPE() { return getToken(dslParser.SHAPE, 0); }
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public ArglistContext arglist() {
			return getRuleContext(ArglistContext.class,0);
		}
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public TerminalNode LBRACE() { return getToken(dslParser.LBRACE, 0); }
		public PropContext prop() {
			return getRuleContext(PropContext.class,0);
		}
		public TerminalNode RBRACE() { return getToken(dslParser.RBRACE, 0); }
		public TerminalNode SEMI() { return getToken(dslParser.SEMI, 0); }
		public Shape_declContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_shape_decl; }
	}

	public final Shape_declContext shape_decl() throws RecognitionException {
		Shape_declContext _localctx = new Shape_declContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_shape_decl);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(48);
			match(SHAPE);
			setState(49);
			match(LPAREN);
			setState(50);
			arglist();
			setState(51);
			match(RPAREN);
			setState(52);
			match(LBRACE);
			setState(53);
			prop(0);
			setState(54);
			match(RBRACE);
			setState(55);
			match(SEMI);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class StatementContext extends ParserRuleContext {
		public StatementContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_statement; }
	 
		public StatementContext() { }
		public void copyFrom(StatementContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class FlowstmtContext extends StatementContext {
		public TerminalNode FLOW() { return getToken(dslParser.FLOW, 0); }
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public DirectionContext direction() {
			return getRuleContext(DirectionContext.class,0);
		}
		public List<TerminalNode> COMMA() { return getTokens(dslParser.COMMA); }
		public TerminalNode COMMA(int i) {
			return getToken(dslParser.COMMA, i);
		}
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode VAR() { return getToken(dslParser.VAR, 0); }
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public TerminalNode SEMI() { return getToken(dslParser.SEMI, 0); }
		public FlowstmtContext(StatementContext ctx) { copyFrom(ctx); }
	}
	public static class FuncstmtContext extends StatementContext {
		public TerminalNode FUNC() { return getToken(dslParser.FUNC, 0); }
		public Func_declContext func_decl() {
			return getRuleContext(Func_declContext.class,0);
		}
		public TerminalNode EQ() { return getToken(dslParser.EQ, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode SEMI() { return getToken(dslParser.SEMI, 0); }
		public FuncstmtContext(StatementContext ctx) { copyFrom(ctx); }
	}
	public static class SeqstmtContext extends StatementContext {
		public List<StatementContext> statement() {
			return getRuleContexts(StatementContext.class);
		}
		public StatementContext statement(int i) {
			return getRuleContext(StatementContext.class,i);
		}
		public SeqstmtContext(StatementContext ctx) { copyFrom(ctx); }
	}
	public static class TransstmtContext extends StatementContext {
		public TransformerContext transformer() {
			return getRuleContext(TransformerContext.class,0);
		}
		public TransstmtContext(StatementContext ctx) { copyFrom(ctx); }
	}

	public final StatementContext statement() throws RecognitionException {
		return statement(0);
	}

	private StatementContext statement(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		StatementContext _localctx = new StatementContext(_ctx, _parentState);
		StatementContext _prevctx = _localctx;
		int _startState = 4;
		enterRecursionRule(_localctx, 4, RULE_statement, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(77);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case FLOW:
				{
				_localctx = new FlowstmtContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(58);
				match(FLOW);
				setState(59);
				match(LPAREN);
				setState(60);
				direction();
				setState(61);
				match(COMMA);
				setState(62);
				expr(0);
				setState(63);
				match(COMMA);
				setState(64);
				expr(0);
				setState(65);
				match(COMMA);
				setState(66);
				match(VAR);
				setState(67);
				match(RPAREN);
				setState(68);
				match(SEMI);
				}
				break;
			case FUNC:
				{
				_localctx = new FuncstmtContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(70);
				match(FUNC);
				setState(71);
				func_decl();
				setState(72);
				match(EQ);
				setState(73);
				expr(0);
				setState(74);
				match(SEMI);
				}
				break;
			case TRANSFORMER:
				{
				_localctx = new TransstmtContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(76);
				transformer();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			_ctx.stop = _input.LT(-1);
			setState(83);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,1,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new SeqstmtContext(new StatementContext(_parentctx, _parentState));
					pushNewRecursionContext(_localctx, _startState, RULE_statement);
					setState(79);
					if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
					setState(80);
					statement(2);
					}
					} 
				}
				setState(85);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,1,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class Func_declContext extends ParserRuleContext {
		public TerminalNode VAR() { return getToken(dslParser.VAR, 0); }
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public ArglistContext arglist() {
			return getRuleContext(ArglistContext.class,0);
		}
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public Func_declContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_func_decl; }
	}

	public final Func_declContext func_decl() throws RecognitionException {
		Func_declContext _localctx = new Func_declContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_func_decl);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(86);
			match(VAR);
			setState(87);
			match(LPAREN);
			setState(88);
			arglist();
			setState(89);
			match(RPAREN);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class TransformerContext extends ParserRuleContext {
		public Trans_declContext trans_decl() {
			return getRuleContext(Trans_declContext.class,0);
		}
		public TerminalNode LBRACE() { return getToken(dslParser.LBRACE, 0); }
		public Op_listContext op_list() {
			return getRuleContext(Op_listContext.class,0);
		}
		public TerminalNode RBRACE() { return getToken(dslParser.RBRACE, 0); }
		public TransformerContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_transformer; }
	}

	public final TransformerContext transformer() throws RecognitionException {
		TransformerContext _localctx = new TransformerContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_transformer);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(91);
			trans_decl();
			setState(92);
			match(LBRACE);
			setState(93);
			op_list();
			setState(94);
			match(RBRACE);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Op_listContext extends ParserRuleContext {
		public Op_stmtContext op_stmt() {
			return getRuleContext(Op_stmtContext.class,0);
		}
		public TerminalNode SEMI() { return getToken(dslParser.SEMI, 0); }
		public Op_listContext op_list() {
			return getRuleContext(Op_listContext.class,0);
		}
		public Op_listContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_op_list; }
	}

	public final Op_listContext op_list() throws RecognitionException {
		Op_listContext _localctx = new Op_listContext(_ctx, getState());
		enterRule(_localctx, 10, RULE_op_list);
		try {
			setState(103);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,2,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(96);
				op_stmt();
				setState(97);
				match(SEMI);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(99);
				op_stmt();
				setState(100);
				match(SEMI);
				setState(101);
				op_list();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Op_stmtContext extends ParserRuleContext {
		public OperatorContext operator() {
			return getRuleContext(OperatorContext.class,0);
		}
		public TerminalNode ARROW() { return getToken(dslParser.ARROW, 0); }
		public Trans_retContext trans_ret() {
			return getRuleContext(Trans_retContext.class,0);
		}
		public Op_stmtContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_op_stmt; }
	}

	public final Op_stmtContext op_stmt() throws RecognitionException {
		Op_stmtContext _localctx = new Op_stmtContext(_ctx, getState());
		enterRule(_localctx, 12, RULE_op_stmt);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(105);
			operator();
			setState(106);
			match(ARROW);
			setState(107);
			trans_ret();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Trans_declContext extends ParserRuleContext {
		public TerminalNode TRANSFORMER() { return getToken(dslParser.TRANSFORMER, 0); }
		public TerminalNode VAR() { return getToken(dslParser.VAR, 0); }
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public TerminalNode CURR() { return getToken(dslParser.CURR, 0); }
		public TerminalNode COMMA() { return getToken(dslParser.COMMA, 0); }
		public TerminalNode PREV() { return getToken(dslParser.PREV, 0); }
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public Trans_declContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_trans_decl; }
	}

	public final Trans_declContext trans_decl() throws RecognitionException {
		Trans_declContext _localctx = new Trans_declContext(_ctx, getState());
		enterRule(_localctx, 14, RULE_trans_decl);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(109);
			match(TRANSFORMER);
			setState(110);
			match(VAR);
			setState(111);
			match(LPAREN);
			setState(112);
			match(CURR);
			setState(113);
			match(COMMA);
			setState(114);
			match(PREV);
			setState(115);
			match(RPAREN);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class OperatorContext extends ParserRuleContext {
		public TerminalNode AFFINE() { return getToken(dslParser.AFFINE, 0); }
		public TerminalNode RELU() { return getToken(dslParser.RELU, 0); }
		public TerminalNode MAXPOOL() { return getToken(dslParser.MAXPOOL, 0); }
		public TerminalNode SIGMOID() { return getToken(dslParser.SIGMOID, 0); }
		public TerminalNode TANH() { return getToken(dslParser.TANH, 0); }
		public OperatorContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_operator; }
	}

	public final OperatorContext operator() throws RecognitionException {
		OperatorContext _localctx = new OperatorContext(_ctx, getState());
		enterRule(_localctx, 16, RULE_operator);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(117);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << AFFINE) | (1L << RELU) | (1L << MAXPOOL) | (1L << SIGMOID) | (1L << TANH))) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Trans_retContext extends ParserRuleContext {
		public Trans_retContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_trans_ret; }
	 
		public Trans_retContext() { }
		public void copyFrom(Trans_retContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class CondtransContext extends Trans_retContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode QUES() { return getToken(dslParser.QUES, 0); }
		public List<Trans_retContext> trans_ret() {
			return getRuleContexts(Trans_retContext.class);
		}
		public Trans_retContext trans_ret(int i) {
			return getRuleContext(Trans_retContext.class,i);
		}
		public TerminalNode COLON() { return getToken(dslParser.COLON, 0); }
		public CondtransContext(Trans_retContext ctx) { copyFrom(ctx); }
	}
	public static class ParentransContext extends Trans_retContext {
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public Trans_retContext trans_ret() {
			return getRuleContext(Trans_retContext.class,0);
		}
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public ParentransContext(Trans_retContext ctx) { copyFrom(ctx); }
	}
	public static class TransContext extends Trans_retContext {
		public Expr_listContext expr_list() {
			return getRuleContext(Expr_listContext.class,0);
		}
		public TransContext(Trans_retContext ctx) { copyFrom(ctx); }
	}

	public final Trans_retContext trans_ret() throws RecognitionException {
		Trans_retContext _localctx = new Trans_retContext(_ctx, getState());
		enterRule(_localctx, 18, RULE_trans_ret);
		try {
			setState(130);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,3,_ctx) ) {
			case 1:
				_localctx = new CondtransContext(_localctx);
				enterOuterAlt(_localctx, 1);
				{
				setState(119);
				expr(0);
				setState(120);
				match(QUES);
				setState(121);
				trans_ret();
				setState(122);
				match(COLON);
				setState(123);
				trans_ret();
				}
				break;
			case 2:
				_localctx = new ParentransContext(_localctx);
				enterOuterAlt(_localctx, 2);
				{
				setState(125);
				match(LPAREN);
				setState(126);
				trans_ret();
				setState(127);
				match(RPAREN);
				}
				break;
			case 3:
				_localctx = new TransContext(_localctx);
				enterOuterAlt(_localctx, 3);
				{
				setState(129);
				expr_list();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class TypesContext extends ParserRuleContext {
		public TerminalNode INTT() { return getToken(dslParser.INTT, 0); }
		public TerminalNode FLOATT() { return getToken(dslParser.FLOATT, 0); }
		public TerminalNode BOOL() { return getToken(dslParser.BOOL, 0); }
		public TerminalNode POLYEXP() { return getToken(dslParser.POLYEXP, 0); }
		public TerminalNode ZONOEXP() { return getToken(dslParser.ZONOEXP, 0); }
		public TerminalNode NEURON() { return getToken(dslParser.NEURON, 0); }
		public TypesContext types() {
			return getRuleContext(TypesContext.class,0);
		}
		public TerminalNode LIST() { return getToken(dslParser.LIST, 0); }
		public TypesContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_types; }
	}

	public final TypesContext types() throws RecognitionException {
		return types(0);
	}

	private TypesContext types(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		TypesContext _localctx = new TypesContext(_ctx, _parentState);
		TypesContext _prevctx = _localctx;
		int _startState = 20;
		enterRecursionRule(_localctx, 20, RULE_types, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(139);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case INTT:
				{
				setState(133);
				match(INTT);
				}
				break;
			case FLOATT:
				{
				setState(134);
				match(FLOATT);
				}
				break;
			case BOOL:
				{
				setState(135);
				match(BOOL);
				}
				break;
			case POLYEXP:
				{
				setState(136);
				match(POLYEXP);
				}
				break;
			case ZONOEXP:
				{
				setState(137);
				match(ZONOEXP);
				}
				break;
			case NEURON:
				{
				setState(138);
				match(NEURON);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			_ctx.stop = _input.LT(-1);
			setState(145);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,5,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new TypesContext(_parentctx, _parentState);
					pushNewRecursionContext(_localctx, _startState, RULE_types);
					setState(141);
					if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
					setState(142);
					match(LIST);
					}
					} 
				}
				setState(147);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,5,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class ArglistContext extends ParserRuleContext {
		public TypesContext types() {
			return getRuleContext(TypesContext.class,0);
		}
		public TerminalNode VAR() { return getToken(dslParser.VAR, 0); }
		public TerminalNode COMMA() { return getToken(dslParser.COMMA, 0); }
		public ArglistContext arglist() {
			return getRuleContext(ArglistContext.class,0);
		}
		public ArglistContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_arglist; }
	}

	public final ArglistContext arglist() throws RecognitionException {
		ArglistContext _localctx = new ArglistContext(_ctx, getState());
		enterRule(_localctx, 22, RULE_arglist);
		try {
			setState(156);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,6,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(148);
				types(0);
				setState(149);
				match(VAR);
				setState(150);
				match(COMMA);
				setState(151);
				arglist();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(153);
				types(0);
				setState(154);
				match(VAR);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Expr_listContext extends ParserRuleContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode COMMA() { return getToken(dslParser.COMMA, 0); }
		public Expr_listContext expr_list() {
			return getRuleContext(Expr_listContext.class,0);
		}
		public Expr_listContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expr_list; }
	}

	public final Expr_listContext expr_list() throws RecognitionException {
		Expr_listContext _localctx = new Expr_listContext(_ctx, getState());
		enterRule(_localctx, 24, RULE_expr_list);
		try {
			setState(163);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,7,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(158);
				expr(0);
				setState(159);
				match(COMMA);
				setState(160);
				expr_list();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(162);
				expr(0);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ExprContext extends ParserRuleContext {
		public ExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expr; }
	 
		public ExprContext() { }
		public void copyFrom(ExprContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class GetMetadataContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode LSQR() { return getToken(dslParser.LSQR, 0); }
		public MetadataContext metadata() {
			return getRuleContext(MetadataContext.class,0);
		}
		public TerminalNode RSQR() { return getToken(dslParser.RSQR, 0); }
		public GetMetadataContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class ArgmaxOpContext extends ExprContext {
		public Argmax_opContext argmax_op() {
			return getRuleContext(Argmax_opContext.class,0);
		}
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode COMMA() { return getToken(dslParser.COMMA, 0); }
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public ArgmaxOpContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class PrevContext extends ExprContext {
		public TerminalNode PREV() { return getToken(dslParser.PREV, 0); }
		public PrevContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class FalseContext extends ExprContext {
		public TerminalNode FALSE() { return getToken(dslParser.FALSE, 0); }
		public FalseContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class MaxOpContext extends ExprContext {
		public Max_opContext max_op() {
			return getRuleContext(Max_opContext.class,0);
		}
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode COMMA() { return getToken(dslParser.COMMA, 0); }
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public MaxOpContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class DotContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode DOT() { return getToken(dslParser.DOT, 0); }
		public TerminalNode DOTT() { return getToken(dslParser.DOTT, 0); }
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public DotContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class FloatContext extends ExprContext {
		public TerminalNode FloatConst() { return getToken(dslParser.FloatConst, 0); }
		public FloatContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class CondContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode QUES() { return getToken(dslParser.QUES, 0); }
		public TerminalNode COLON() { return getToken(dslParser.COLON, 0); }
		public CondContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class IntContext extends ExprContext {
		public TerminalNode IntConst() { return getToken(dslParser.IntConst, 0); }
		public IntContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class EpsilonContext extends ExprContext {
		public TerminalNode EPSILON() { return getToken(dslParser.EPSILON, 0); }
		public EpsilonContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class VarExpContext extends ExprContext {
		public TerminalNode VAR() { return getToken(dslParser.VAR, 0); }
		public VarExpContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class NegContext extends ExprContext {
		public TerminalNode MINUS() { return getToken(dslParser.MINUS, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public NegContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class TraverseContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode DOT() { return getToken(dslParser.DOT, 0); }
		public TerminalNode TRAV() { return getToken(dslParser.TRAV, 0); }
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public DirectionContext direction() {
			return getRuleContext(DirectionContext.class,0);
		}
		public List<TerminalNode> COMMA() { return getTokens(dslParser.COMMA); }
		public TerminalNode COMMA(int i) {
			return getToken(dslParser.COMMA, i);
		}
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public TerminalNode LBRACE() { return getToken(dslParser.LBRACE, 0); }
		public PropContext prop() {
			return getRuleContext(PropContext.class,0);
		}
		public TerminalNode RBRACE() { return getToken(dslParser.RBRACE, 0); }
		public TraverseContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class NotContext extends ExprContext {
		public TerminalNode NOT() { return getToken(dslParser.NOT, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public NotContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class BinopExpContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public BinopContext binop() {
			return getRuleContext(BinopContext.class,0);
		}
		public BinopExpContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class ListOpContext extends ExprContext {
		public List_opContext list_op() {
			return getRuleContext(List_opContext.class,0);
		}
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public ListOpContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class GetElementContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode LSQR() { return getToken(dslParser.LSQR, 0); }
		public TerminalNode VAR() { return getToken(dslParser.VAR, 0); }
		public TerminalNode RSQR() { return getToken(dslParser.RSQR, 0); }
		public GetElementContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class TrueContext extends ExprContext {
		public TerminalNode TRUE() { return getToken(dslParser.TRUE, 0); }
		public TrueContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class ParenExpContext extends ExprContext {
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public ParenExpContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class FuncCallContext extends ExprContext {
		public TerminalNode VAR() { return getToken(dslParser.VAR, 0); }
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public Expr_listContext expr_list() {
			return getRuleContext(Expr_listContext.class,0);
		}
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public FuncCallContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class CurrContext extends ExprContext {
		public TerminalNode CURR() { return getToken(dslParser.CURR, 0); }
		public CurrContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class MaxOpListContext extends ExprContext {
		public Max_opContext max_op() {
			return getRuleContext(Max_opContext.class,0);
		}
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public MaxOpListContext(ExprContext ctx) { copyFrom(ctx); }
	}
	public static class MapContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode DOT() { return getToken(dslParser.DOT, 0); }
		public TerminalNode MAP() { return getToken(dslParser.MAP, 0); }
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public MapContext(ExprContext ctx) { copyFrom(ctx); }
	}

	public final ExprContext expr() throws RecognitionException {
		return expr(0);
	}

	private ExprContext expr(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ExprContext _localctx = new ExprContext(_ctx, _parentState);
		ExprContext _prevctx = _localctx;
		int _startState = 26;
		enterRecursionRule(_localctx, 26, RULE_expr, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(211);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,8,_ctx) ) {
			case 1:
				{
				_localctx = new FalseContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(166);
				match(FALSE);
				}
				break;
			case 2:
				{
				_localctx = new TrueContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(167);
				match(TRUE);
				}
				break;
			case 3:
				{
				_localctx = new IntContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(168);
				match(IntConst);
				}
				break;
			case 4:
				{
				_localctx = new FloatContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(169);
				match(FloatConst);
				}
				break;
			case 5:
				{
				_localctx = new VarExpContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(170);
				match(VAR);
				}
				break;
			case 6:
				{
				_localctx = new EpsilonContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(171);
				match(EPSILON);
				}
				break;
			case 7:
				{
				_localctx = new CurrContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(172);
				match(CURR);
				}
				break;
			case 8:
				{
				_localctx = new PrevContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(173);
				match(PREV);
				}
				break;
			case 9:
				{
				_localctx = new ParenExpContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(174);
				match(LPAREN);
				setState(175);
				expr(0);
				setState(176);
				match(RPAREN);
				}
				break;
			case 10:
				{
				_localctx = new NotContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(178);
				match(NOT);
				setState(179);
				expr(11);
				}
				break;
			case 11:
				{
				_localctx = new NegContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(180);
				match(MINUS);
				setState(181);
				expr(10);
				}
				break;
			case 12:
				{
				_localctx = new ArgmaxOpContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(182);
				argmax_op();
				setState(183);
				match(LPAREN);
				setState(184);
				expr(0);
				setState(185);
				match(COMMA);
				setState(186);
				expr(0);
				setState(187);
				match(RPAREN);
				}
				break;
			case 13:
				{
				_localctx = new MaxOpListContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(189);
				max_op();
				setState(190);
				match(LPAREN);
				setState(191);
				expr(0);
				setState(192);
				match(RPAREN);
				}
				break;
			case 14:
				{
				_localctx = new MaxOpContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(194);
				max_op();
				setState(195);
				match(LPAREN);
				setState(196);
				expr(0);
				setState(197);
				match(COMMA);
				setState(198);
				expr(0);
				setState(199);
				match(RPAREN);
				}
				break;
			case 15:
				{
				_localctx = new ListOpContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(201);
				list_op();
				setState(202);
				match(LPAREN);
				setState(203);
				expr(0);
				setState(204);
				match(RPAREN);
				}
				break;
			case 16:
				{
				_localctx = new FuncCallContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(206);
				match(VAR);
				setState(207);
				match(LPAREN);
				setState(208);
				expr_list();
				setState(209);
				match(RPAREN);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(264);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,10,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(262);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,9,_ctx) ) {
					case 1:
						{
						_localctx = new BinopExpContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(213);
						if (!(precpred(_ctx, 12))) throw new FailedPredicateException(this, "precpred(_ctx, 12)");
						setState(214);
						binop();
						setState(215);
						expr(13);
						}
						break;
					case 2:
						{
						_localctx = new CondContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(217);
						if (!(precpred(_ctx, 9))) throw new FailedPredicateException(this, "precpred(_ctx, 9)");
						setState(218);
						match(QUES);
						setState(219);
						expr(0);
						setState(220);
						match(COLON);
						setState(221);
						expr(10);
						}
						break;
					case 3:
						{
						_localctx = new GetMetadataContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(223);
						if (!(precpred(_ctx, 14))) throw new FailedPredicateException(this, "precpred(_ctx, 14)");
						setState(224);
						match(LSQR);
						setState(225);
						metadata();
						setState(226);
						match(RSQR);
						}
						break;
					case 4:
						{
						_localctx = new GetElementContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(228);
						if (!(precpred(_ctx, 13))) throw new FailedPredicateException(this, "precpred(_ctx, 13)");
						setState(229);
						match(LSQR);
						setState(230);
						match(VAR);
						setState(231);
						match(RSQR);
						}
						break;
					case 5:
						{
						_localctx = new TraverseContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(232);
						if (!(precpred(_ctx, 8))) throw new FailedPredicateException(this, "precpred(_ctx, 8)");
						setState(233);
						match(DOT);
						setState(234);
						match(TRAV);
						setState(235);
						match(LPAREN);
						setState(236);
						direction();
						setState(237);
						match(COMMA);
						setState(238);
						expr(0);
						setState(239);
						match(COMMA);
						setState(240);
						expr(0);
						setState(241);
						match(COMMA);
						setState(242);
						expr(0);
						setState(243);
						match(RPAREN);
						setState(244);
						match(LBRACE);
						setState(245);
						prop(0);
						setState(246);
						match(RBRACE);
						}
						break;
					case 6:
						{
						_localctx = new MapContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(248);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(249);
						match(DOT);
						setState(250);
						match(MAP);
						setState(251);
						match(LPAREN);
						setState(252);
						expr(0);
						setState(253);
						match(RPAREN);
						}
						break;
					case 7:
						{
						_localctx = new DotContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(255);
						if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
						setState(256);
						match(DOT);
						setState(257);
						match(DOTT);
						setState(258);
						match(LPAREN);
						setState(259);
						expr(0);
						setState(260);
						match(RPAREN);
						}
						break;
					}
					} 
				}
				setState(266);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,10,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class Argmax_opContext extends ParserRuleContext {
		public TerminalNode ARGMAX() { return getToken(dslParser.ARGMAX, 0); }
		public TerminalNode ARGMIN() { return getToken(dslParser.ARGMIN, 0); }
		public Argmax_opContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_argmax_op; }
	}

	public final Argmax_opContext argmax_op() throws RecognitionException {
		Argmax_opContext _localctx = new Argmax_opContext(_ctx, getState());
		enterRule(_localctx, 28, RULE_argmax_op);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(267);
			_la = _input.LA(1);
			if ( !(_la==ARGMIN || _la==ARGMAX) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Max_opContext extends ParserRuleContext {
		public TerminalNode MAX() { return getToken(dslParser.MAX, 0); }
		public TerminalNode MIN() { return getToken(dslParser.MIN, 0); }
		public Max_opContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_max_op; }
	}

	public final Max_opContext max_op() throws RecognitionException {
		Max_opContext _localctx = new Max_opContext(_ctx, getState());
		enterRule(_localctx, 30, RULE_max_op);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(269);
			_la = _input.LA(1);
			if ( !(_la==MIN || _la==MAX) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class List_opContext extends ParserRuleContext {
		public TerminalNode SUM() { return getToken(dslParser.SUM, 0); }
		public TerminalNode LEN() { return getToken(dslParser.LEN, 0); }
		public TerminalNode AVG() { return getToken(dslParser.AVG, 0); }
		public List_opContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_list_op; }
	}

	public final List_opContext list_op() throws RecognitionException {
		List_opContext _localctx = new List_opContext(_ctx, getState());
		enterRule(_localctx, 32, RULE_list_op);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(271);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << SUM) | (1L << LEN) | (1L << AVG))) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class BinopContext extends ParserRuleContext {
		public TerminalNode PLUS() { return getToken(dslParser.PLUS, 0); }
		public TerminalNode MINUS() { return getToken(dslParser.MINUS, 0); }
		public TerminalNode MULT() { return getToken(dslParser.MULT, 0); }
		public TerminalNode DIV() { return getToken(dslParser.DIV, 0); }
		public TerminalNode AND() { return getToken(dslParser.AND, 0); }
		public TerminalNode OR() { return getToken(dslParser.OR, 0); }
		public TerminalNode GEQ() { return getToken(dslParser.GEQ, 0); }
		public TerminalNode LEQ() { return getToken(dslParser.LEQ, 0); }
		public TerminalNode LT() { return getToken(dslParser.LT, 0); }
		public TerminalNode GT() { return getToken(dslParser.GT, 0); }
		public TerminalNode EQQ() { return getToken(dslParser.EQQ, 0); }
		public BinopContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_binop; }
	}

	public final BinopContext binop() throws RecognitionException {
		BinopContext _localctx = new BinopContext(_ctx, getState());
		enterRule(_localctx, 34, RULE_binop);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(273);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << PLUS) | (1L << MINUS) | (1L << MULT) | (1L << DIV) | (1L << AND) | (1L << OR) | (1L << LT) | (1L << EQQ) | (1L << GT) | (1L << LEQ) | (1L << GEQ))) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class MetadataContext extends ParserRuleContext {
		public TerminalNode WEIGHT() { return getToken(dslParser.WEIGHT, 0); }
		public TerminalNode BIAS() { return getToken(dslParser.BIAS, 0); }
		public TerminalNode LAYER() { return getToken(dslParser.LAYER, 0); }
		public MetadataContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_metadata; }
	}

	public final MetadataContext metadata() throws RecognitionException {
		MetadataContext _localctx = new MetadataContext(_ctx, getState());
		enterRule(_localctx, 36, RULE_metadata);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(275);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << WEIGHT) | (1L << BIAS) | (1L << LAYER))) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class DirectionContext extends ParserRuleContext {
		public TerminalNode BACKWARD() { return getToken(dslParser.BACKWARD, 0); }
		public TerminalNode FORWARD() { return getToken(dslParser.FORWARD, 0); }
		public DirectionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_direction; }
	}

	public final DirectionContext direction() throws RecognitionException {
		DirectionContext _localctx = new DirectionContext(_ctx, getState());
		enterRule(_localctx, 38, RULE_direction);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(277);
			_la = _input.LA(1);
			if ( !(_la==BACKWARD || _la==FORWARD) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class PtContext extends ParserRuleContext {
		public PtContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_pt; }
	 
		public PtContext() { }
		public void copyFrom(PtContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class PtopContext extends PtContext {
		public List<PtContext> pt() {
			return getRuleContexts(PtContext.class);
		}
		public PtContext pt(int i) {
			return getRuleContext(PtContext.class,i);
		}
		public TerminalNode PLUS() { return getToken(dslParser.PLUS, 0); }
		public TerminalNode MINUS() { return getToken(dslParser.MINUS, 0); }
		public PtopContext(PtContext ctx) { copyFrom(ctx); }
	}
	public static class PtbasicContext extends PtContext {
		public TerminalNode IntConst() { return getToken(dslParser.IntConst, 0); }
		public TerminalNode FloatConst() { return getToken(dslParser.FloatConst, 0); }
		public TerminalNode FALSE() { return getToken(dslParser.FALSE, 0); }
		public TerminalNode TRUE() { return getToken(dslParser.TRUE, 0); }
		public TerminalNode VAR() { return getToken(dslParser.VAR, 0); }
		public TerminalNode CURR() { return getToken(dslParser.CURR, 0); }
		public PtContext pt() {
			return getRuleContext(PtContext.class,0);
		}
		public TerminalNode LSQR() { return getToken(dslParser.LSQR, 0); }
		public TerminalNode RSQR() { return getToken(dslParser.RSQR, 0); }
		public MetadataContext metadata() {
			return getRuleContext(MetadataContext.class,0);
		}
		public PtbasicContext(PtContext ctx) { copyFrom(ctx); }
	}

	public final PtContext pt() throws RecognitionException {
		return pt(0);
	}

	private PtContext pt(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		PtContext _localctx = new PtContext(_ctx, _parentState);
		PtContext _prevctx = _localctx;
		int _startState = 40;
		enterRecursionRule(_localctx, 40, RULE_pt, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(286);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case IntConst:
				{
				_localctx = new PtbasicContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(280);
				match(IntConst);
				}
				break;
			case FloatConst:
				{
				_localctx = new PtbasicContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(281);
				match(FloatConst);
				}
				break;
			case FALSE:
				{
				_localctx = new PtbasicContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(282);
				match(FALSE);
				}
				break;
			case TRUE:
				{
				_localctx = new PtbasicContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(283);
				match(TRUE);
				}
				break;
			case VAR:
				{
				_localctx = new PtbasicContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(284);
				match(VAR);
				}
				break;
			case CURR:
				{
				_localctx = new PtbasicContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(285);
				match(CURR);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			_ctx.stop = _input.LT(-1);
			setState(305);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,13,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(303);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,12,_ctx) ) {
					case 1:
						{
						_localctx = new PtopContext(new PtContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_pt);
						setState(288);
						if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
						setState(289);
						match(PLUS);
						setState(290);
						pt(3);
						}
						break;
					case 2:
						{
						_localctx = new PtopContext(new PtContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_pt);
						setState(291);
						if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
						setState(292);
						match(MINUS);
						setState(293);
						pt(2);
						}
						break;
					case 3:
						{
						_localctx = new PtbasicContext(new PtContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_pt);
						setState(294);
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(295);
						match(LSQR);
						setState(296);
						match(VAR);
						setState(297);
						match(RSQR);
						}
						break;
					case 4:
						{
						_localctx = new PtbasicContext(new PtContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_pt);
						setState(298);
						if (!(precpred(_ctx, 5))) throw new FailedPredicateException(this, "precpred(_ctx, 5)");
						setState(299);
						match(LSQR);
						setState(300);
						metadata();
						setState(301);
						match(RSQR);
						}
						break;
					}
					} 
				}
				setState(307);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,13,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class PropContext extends ParserRuleContext {
		public PropContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_prop; }
	 
		public PropContext() { }
		public void copyFrom(PropContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class PropsingleContext extends PropContext {
		public List<PtContext> pt() {
			return getRuleContexts(PtContext.class);
		}
		public PtContext pt(int i) {
			return getRuleContext(PtContext.class,i);
		}
		public TerminalNode GT() { return getToken(dslParser.GT, 0); }
		public TerminalNode GEQ() { return getToken(dslParser.GEQ, 0); }
		public TerminalNode LEQ() { return getToken(dslParser.LEQ, 0); }
		public TerminalNode LT() { return getToken(dslParser.LT, 0); }
		public TerminalNode EQQ() { return getToken(dslParser.EQQ, 0); }
		public PropsingleContext(PropContext ctx) { copyFrom(ctx); }
	}
	public static class PropparenContext extends PropContext {
		public TerminalNode LPAREN() { return getToken(dslParser.LPAREN, 0); }
		public PropContext prop() {
			return getRuleContext(PropContext.class,0);
		}
		public TerminalNode RPAREN() { return getToken(dslParser.RPAREN, 0); }
		public PropparenContext(PropContext ctx) { copyFrom(ctx); }
	}
	public static class PropdoubleContext extends PropContext {
		public List<PropContext> prop() {
			return getRuleContexts(PropContext.class);
		}
		public PropContext prop(int i) {
			return getRuleContext(PropContext.class,i);
		}
		public TerminalNode AND() { return getToken(dslParser.AND, 0); }
		public TerminalNode OR() { return getToken(dslParser.OR, 0); }
		public PropdoubleContext(PropContext ctx) { copyFrom(ctx); }
	}
	public static class PtinContext extends PropContext {
		public List<PtContext> pt() {
			return getRuleContexts(PtContext.class);
		}
		public PtContext pt(int i) {
			return getRuleContext(PtContext.class,i);
		}
		public TerminalNode IN() { return getToken(dslParser.IN, 0); }
		public PtinContext(PropContext ctx) { copyFrom(ctx); }
	}

	public final PropContext prop() throws RecognitionException {
		return prop(0);
	}

	private PropContext prop(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		PropContext _localctx = new PropContext(_ctx, _parentState);
		PropContext _prevctx = _localctx;
		int _startState = 42;
		enterRecursionRule(_localctx, 42, RULE_prop, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(337);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,14,_ctx) ) {
			case 1:
				{
				_localctx = new PropparenContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(309);
				match(LPAREN);
				setState(310);
				prop(0);
				setState(311);
				match(RPAREN);
				}
				break;
			case 2:
				{
				_localctx = new PropsingleContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(313);
				pt(0);
				setState(314);
				match(GT);
				setState(315);
				pt(0);
				}
				break;
			case 3:
				{
				_localctx = new PropsingleContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(317);
				pt(0);
				setState(318);
				match(GEQ);
				setState(319);
				pt(0);
				}
				break;
			case 4:
				{
				_localctx = new PropsingleContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(321);
				pt(0);
				setState(322);
				match(LEQ);
				setState(323);
				pt(0);
				}
				break;
			case 5:
				{
				_localctx = new PropsingleContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(325);
				pt(0);
				setState(326);
				match(LT);
				setState(327);
				pt(0);
				}
				break;
			case 6:
				{
				_localctx = new PropsingleContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(329);
				pt(0);
				setState(330);
				match(EQQ);
				setState(331);
				pt(0);
				}
				break;
			case 7:
				{
				_localctx = new PtinContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(333);
				pt(0);
				setState(334);
				match(IN);
				setState(335);
				pt(0);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(347);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,16,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(345);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,15,_ctx) ) {
					case 1:
						{
						_localctx = new PropdoubleContext(new PropContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_prop);
						setState(339);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(340);
						match(AND);
						setState(341);
						prop(4);
						}
						break;
					case 2:
						{
						_localctx = new PropdoubleContext(new PropContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_prop);
						setState(342);
						if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
						setState(343);
						match(OR);
						setState(344);
						prop(3);
						}
						break;
					}
					} 
				}
				setState(349);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,16,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 2:
			return statement_sempred((StatementContext)_localctx, predIndex);
		case 10:
			return types_sempred((TypesContext)_localctx, predIndex);
		case 13:
			return expr_sempred((ExprContext)_localctx, predIndex);
		case 20:
			return pt_sempred((PtContext)_localctx, predIndex);
		case 21:
			return prop_sempred((PropContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean statement_sempred(StatementContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 1);
		}
		return true;
	}
	private boolean types_sempred(TypesContext _localctx, int predIndex) {
		switch (predIndex) {
		case 1:
			return precpred(_ctx, 1);
		}
		return true;
	}
	private boolean expr_sempred(ExprContext _localctx, int predIndex) {
		switch (predIndex) {
		case 2:
			return precpred(_ctx, 12);
		case 3:
			return precpred(_ctx, 9);
		case 4:
			return precpred(_ctx, 14);
		case 5:
			return precpred(_ctx, 13);
		case 6:
			return precpred(_ctx, 8);
		case 7:
			return precpred(_ctx, 3);
		case 8:
			return precpred(_ctx, 2);
		}
		return true;
	}
	private boolean pt_sempred(PtContext _localctx, int predIndex) {
		switch (predIndex) {
		case 9:
			return precpred(_ctx, 2);
		case 10:
			return precpred(_ctx, 1);
		case 11:
			return precpred(_ctx, 6);
		case 12:
			return precpred(_ctx, 5);
		}
		return true;
	}
	private boolean prop_sempred(PropContext _localctx, int predIndex) {
		switch (predIndex) {
		case 13:
			return precpred(_ctx, 3);
		case 14:
			return precpred(_ctx, 2);
		}
		return true;
	}

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3I\u0161\4\2\t\2\4"+
		"\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t"+
		"\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22"+
		"\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\3\2\3\2\3\2\3\2\3\3"+
		"\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3"+
		"\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\5\4P\n\4\3\4\3\4\7\4T\n\4\f"+
		"\4\16\4W\13\4\3\5\3\5\3\5\3\5\3\5\3\6\3\6\3\6\3\6\3\6\3\7\3\7\3\7\3\7"+
		"\3\7\3\7\3\7\5\7j\n\7\3\b\3\b\3\b\3\b\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t"+
		"\3\n\3\n\3\13\3\13\3\13\3\13\3\13\3\13\3\13\3\13\3\13\3\13\3\13\5\13\u0085"+
		"\n\13\3\f\3\f\3\f\3\f\3\f\3\f\3\f\5\f\u008e\n\f\3\f\3\f\7\f\u0092\n\f"+
		"\f\f\16\f\u0095\13\f\3\r\3\r\3\r\3\r\3\r\3\r\3\r\3\r\5\r\u009f\n\r\3\16"+
		"\3\16\3\16\3\16\3\16\5\16\u00a6\n\16\3\17\3\17\3\17\3\17\3\17\3\17\3\17"+
		"\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17"+
		"\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17"+
		"\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\5\17\u00d6\n\17"+
		"\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17"+
		"\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17"+
		"\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17"+
		"\3\17\3\17\3\17\3\17\3\17\3\17\3\17\7\17\u0109\n\17\f\17\16\17\u010c\13"+
		"\17\3\20\3\20\3\21\3\21\3\22\3\22\3\23\3\23\3\24\3\24\3\25\3\25\3\26\3"+
		"\26\3\26\3\26\3\26\3\26\3\26\5\26\u0121\n\26\3\26\3\26\3\26\3\26\3\26"+
		"\3\26\3\26\3\26\3\26\3\26\3\26\3\26\3\26\3\26\3\26\7\26\u0132\n\26\f\26"+
		"\16\26\u0135\13\26\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3"+
		"\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3"+
		"\27\3\27\3\27\3\27\3\27\5\27\u0154\n\27\3\27\3\27\3\27\3\27\3\27\3\27"+
		"\7\27\u015c\n\27\f\27\16\27\u015f\13\27\3\27\2\7\6\26\34*,\30\2\4\6\b"+
		"\n\f\16\20\22\24\26\30\32\34\36 \"$&(*,\2\t\3\29=\3\2\62\63\3\2\64\65"+
		"\3\2,.\5\2\23\31\33\33\35\37\3\2\668\3\2\b\t\2\u017f\2.\3\2\2\2\4\62\3"+
		"\2\2\2\6O\3\2\2\2\bX\3\2\2\2\n]\3\2\2\2\fi\3\2\2\2\16k\3\2\2\2\20o\3\2"+
		"\2\2\22w\3\2\2\2\24\u0084\3\2\2\2\26\u008d\3\2\2\2\30\u009e\3\2\2\2\32"+
		"\u00a5\3\2\2\2\34\u00d5\3\2\2\2\36\u010d\3\2\2\2 \u010f\3\2\2\2\"\u0111"+
		"\3\2\2\2$\u0113\3\2\2\2&\u0115\3\2\2\2(\u0117\3\2\2\2*\u0120\3\2\2\2,"+
		"\u0153\3\2\2\2./\5\4\3\2/\60\5\6\4\2\60\61\7\2\2\3\61\3\3\2\2\2\62\63"+
		"\7>\2\2\63\64\7!\2\2\64\65\5\30\r\2\65\66\7\"\2\2\66\67\7%\2\2\678\5,"+
		"\27\289\7&\2\29:\7\'\2\2:\5\3\2\2\2;<\b\4\1\2<=\7\3\2\2=>\7!\2\2>?\5("+
		"\25\2?@\7\22\2\2@A\5\34\17\2AB\7\22\2\2BC\5\34\17\2CD\7\22\2\2DE\7G\2"+
		"\2EF\7\"\2\2FG\7\'\2\2GP\3\2\2\2HI\7?\2\2IJ\5\b\5\2JK\7\32\2\2KL\5\34"+
		"\17\2LM\7\'\2\2MP\3\2\2\2NP\5\n\6\2O;\3\2\2\2OH\3\2\2\2ON\3\2\2\2PU\3"+
		"\2\2\2QR\f\3\2\2RT\5\6\4\4SQ\3\2\2\2TW\3\2\2\2US\3\2\2\2UV\3\2\2\2V\7"+
		"\3\2\2\2WU\3\2\2\2XY\7G\2\2YZ\7!\2\2Z[\5\30\r\2[\\\7\"\2\2\\\t\3\2\2\2"+
		"]^\5\20\t\2^_\7%\2\2_`\5\f\7\2`a\7&\2\2a\13\3\2\2\2bc\5\16\b\2cd\7\'\2"+
		"\2dj\3\2\2\2ef\5\16\b\2fg\7\'\2\2gh\5\f\7\2hj\3\2\2\2ib\3\2\2\2ie\3\2"+
		"\2\2j\r\3\2\2\2kl\5\22\n\2lm\7\4\2\2mn\5\24\13\2n\17\3\2\2\2op\7\5\2\2"+
		"pq\7G\2\2qr\7!\2\2rs\7C\2\2st\7\22\2\2tu\7D\2\2uv\7\"\2\2v\21\3\2\2\2"+
		"wx\t\2\2\2x\23\3\2\2\2yz\5\34\17\2z{\7(\2\2{|\5\24\13\2|}\7)\2\2}~\5\24"+
		"\13\2~\u0085\3\2\2\2\177\u0080\7!\2\2\u0080\u0081\5\24\13\2\u0081\u0082"+
		"\7\"\2\2\u0082\u0085\3\2\2\2\u0083\u0085\5\32\16\2\u0084y\3\2\2\2\u0084"+
		"\177\3\2\2\2\u0084\u0083\3\2\2\2\u0085\25\3\2\2\2\u0086\u0087\b\f\1\2"+
		"\u0087\u008e\7\n\2\2\u0088\u008e\7\13\2\2\u0089\u008e\7\f\2\2\u008a\u008e"+
		"\7\r\2\2\u008b\u008e\7\16\2\2\u008c\u008e\7\17\2\2\u008d\u0086\3\2\2\2"+
		"\u008d\u0088\3\2\2\2\u008d\u0089\3\2\2\2\u008d\u008a\3\2\2\2\u008d\u008b"+
		"\3\2\2\2\u008d\u008c\3\2\2\2\u008e\u0093\3\2\2\2\u008f\u0090\f\3\2\2\u0090"+
		"\u0092\7\20\2\2\u0091\u008f\3\2\2\2\u0092\u0095\3\2\2\2\u0093\u0091\3"+
		"\2\2\2\u0093\u0094\3\2\2\2\u0094\27\3\2\2\2\u0095\u0093\3\2\2\2\u0096"+
		"\u0097\5\26\f\2\u0097\u0098\7G\2\2\u0098\u0099\7\22\2\2\u0099\u009a\5"+
		"\30\r\2\u009a\u009f\3\2\2\2\u009b\u009c\5\26\f\2\u009c\u009d\7G\2\2\u009d"+
		"\u009f\3\2\2\2\u009e\u0096\3\2\2\2\u009e\u009b\3\2\2\2\u009f\31\3\2\2"+
		"\2\u00a0\u00a1\5\34\17\2\u00a1\u00a2\7\22\2\2\u00a2\u00a3\5\32\16\2\u00a3"+
		"\u00a6\3\2\2\2\u00a4\u00a6\5\34\17\2\u00a5\u00a0\3\2\2\2\u00a5\u00a4\3"+
		"\2\2\2\u00a6\33\3\2\2\2\u00a7\u00a8\b\17\1\2\u00a8\u00d6\7B\2\2\u00a9"+
		"\u00d6\7A\2\2\u00aa\u00d6\7E\2\2\u00ab\u00d6\7F\2\2\u00ac\u00d6\7G\2\2"+
		"\u00ad\u00d6\7@\2\2\u00ae\u00d6\7C\2\2\u00af\u00d6\7D\2\2\u00b0\u00b1"+
		"\7!\2\2\u00b1\u00b2\5\34\17\2\u00b2\u00b3\7\"\2\2\u00b3\u00d6\3\2\2\2"+
		"\u00b4\u00b5\7 \2\2\u00b5\u00d6\5\34\17\r\u00b6\u00b7\7\24\2\2\u00b7\u00d6"+
		"\5\34\17\f\u00b8\u00b9\5\36\20\2\u00b9\u00ba\7!\2\2\u00ba\u00bb\5\34\17"+
		"\2\u00bb\u00bc\7\22\2\2\u00bc\u00bd\5\34\17\2\u00bd\u00be\7\"\2\2\u00be"+
		"\u00d6\3\2\2\2\u00bf\u00c0\5 \21\2\u00c0\u00c1\7!\2\2\u00c1\u00c2\5\34"+
		"\17\2\u00c2\u00c3\7\"\2\2\u00c3\u00d6\3\2\2\2\u00c4\u00c5\5 \21\2\u00c5"+
		"\u00c6\7!\2\2\u00c6\u00c7\5\34\17\2\u00c7\u00c8\7\22\2\2\u00c8\u00c9\5"+
		"\34\17\2\u00c9\u00ca\7\"\2\2\u00ca\u00d6\3\2\2\2\u00cb\u00cc\5\"\22\2"+
		"\u00cc\u00cd\7!\2\2\u00cd\u00ce\5\34\17\2\u00ce\u00cf\7\"\2\2\u00cf\u00d6"+
		"\3\2\2\2\u00d0\u00d1\7G\2\2\u00d1\u00d2\7!\2\2\u00d2\u00d3\5\32\16\2\u00d3"+
		"\u00d4\7\"\2\2\u00d4\u00d6\3\2\2\2\u00d5\u00a7\3\2\2\2\u00d5\u00a9\3\2"+
		"\2\2\u00d5\u00aa\3\2\2\2\u00d5\u00ab\3\2\2\2\u00d5\u00ac\3\2\2\2\u00d5"+
		"\u00ad\3\2\2\2\u00d5\u00ae\3\2\2\2\u00d5\u00af\3\2\2\2\u00d5\u00b0\3\2"+
		"\2\2\u00d5\u00b4\3\2\2\2\u00d5\u00b6\3\2\2\2\u00d5\u00b8\3\2\2\2\u00d5"+
		"\u00bf\3\2\2\2\u00d5\u00c4\3\2\2\2\u00d5\u00cb\3\2\2\2\u00d5\u00d0\3\2"+
		"\2\2\u00d6\u010a\3\2\2\2\u00d7\u00d8\f\16\2\2\u00d8\u00d9\5$\23\2\u00d9"+
		"\u00da\5\34\17\17\u00da\u0109\3\2\2\2\u00db\u00dc\f\13\2\2\u00dc\u00dd"+
		"\7(\2\2\u00dd\u00de\5\34\17\2\u00de\u00df\7)\2\2\u00df\u00e0\5\34\17\f"+
		"\u00e0\u0109\3\2\2\2\u00e1\u00e2\f\20\2\2\u00e2\u00e3\7#\2\2\u00e3\u00e4"+
		"\5&\24\2\u00e4\u00e5\7$\2\2\u00e5\u0109\3\2\2\2\u00e6\u00e7\f\17\2\2\u00e7"+
		"\u00e8\7#\2\2\u00e8\u00e9\7G\2\2\u00e9\u0109\7$\2\2\u00ea\u00eb\f\n\2"+
		"\2\u00eb\u00ec\7\21\2\2\u00ec\u00ed\7+\2\2\u00ed\u00ee\7!\2\2\u00ee\u00ef"+
		"\5(\25\2\u00ef\u00f0\7\22\2\2\u00f0\u00f1\5\34\17\2\u00f1\u00f2\7\22\2"+
		"\2\u00f2\u00f3\5\34\17\2\u00f3\u00f4\7\22\2\2\u00f4\u00f5\5\34\17\2\u00f5"+
		"\u00f6\7\"\2\2\u00f6\u00f7\7%\2\2\u00f7\u00f8\5,\27\2\u00f8\u00f9\7&\2"+
		"\2\u00f9\u0109\3\2\2\2\u00fa\u00fb\f\5\2\2\u00fb\u00fc\7\21\2\2\u00fc"+
		"\u00fd\7\60\2\2\u00fd\u00fe\7!\2\2\u00fe\u00ff\5\34\17\2\u00ff\u0100\7"+
		"\"\2\2\u0100\u0109\3\2\2\2\u0101\u0102\f\4\2\2\u0102\u0103\7\21\2\2\u0103"+
		"\u0104\7\61\2\2\u0104\u0105\7!\2\2\u0105\u0106\5\34\17\2\u0106\u0107\7"+
		"\"\2\2\u0107\u0109\3\2\2\2\u0108\u00d7\3\2\2\2\u0108\u00db\3\2\2\2\u0108"+
		"\u00e1\3\2\2\2\u0108\u00e6\3\2\2\2\u0108\u00ea\3\2\2\2\u0108\u00fa\3\2"+
		"\2\2\u0108\u0101\3\2\2\2\u0109\u010c\3\2\2\2\u010a\u0108\3\2\2\2\u010a"+
		"\u010b\3\2\2\2\u010b\35\3\2\2\2\u010c\u010a\3\2\2\2\u010d\u010e\t\3\2"+
		"\2\u010e\37\3\2\2\2\u010f\u0110\t\4\2\2\u0110!\3\2\2\2\u0111\u0112\t\5"+
		"\2\2\u0112#\3\2\2\2\u0113\u0114\t\6\2\2\u0114%\3\2\2\2\u0115\u0116\t\7"+
		"\2\2\u0116\'\3\2\2\2\u0117\u0118\t\b\2\2\u0118)\3\2\2\2\u0119\u011a\b"+
		"\26\1\2\u011a\u0121\7E\2\2\u011b\u0121\7F\2\2\u011c\u0121\7B\2\2\u011d"+
		"\u0121\7A\2\2\u011e\u0121\7G\2\2\u011f\u0121\7C\2\2\u0120\u0119\3\2\2"+
		"\2\u0120\u011b\3\2\2\2\u0120\u011c\3\2\2\2\u0120\u011d\3\2\2\2\u0120\u011e"+
		"\3\2\2\2\u0120\u011f\3\2\2\2\u0121\u0133\3\2\2\2\u0122\u0123\f\4\2\2\u0123"+
		"\u0124\7\23\2\2\u0124\u0132\5*\26\5\u0125\u0126\f\3\2\2\u0126\u0127\7"+
		"\24\2\2\u0127\u0132\5*\26\4\u0128\u0129\f\b\2\2\u0129\u012a\7#\2\2\u012a"+
		"\u012b\7G\2\2\u012b\u0132\7$\2\2\u012c\u012d\f\7\2\2\u012d\u012e\7#\2"+
		"\2\u012e\u012f\5&\24\2\u012f\u0130\7$\2\2\u0130\u0132\3\2\2\2\u0131\u0122"+
		"\3\2\2\2\u0131\u0125\3\2\2\2\u0131\u0128\3\2\2\2\u0131\u012c\3\2\2\2\u0132"+
		"\u0135\3\2\2\2\u0133\u0131\3\2\2\2\u0133\u0134\3\2\2\2\u0134+\3\2\2\2"+
		"\u0135\u0133\3\2\2\2\u0136\u0137\b\27\1\2\u0137\u0138\7!\2\2\u0138\u0139"+
		"\5,\27\2\u0139\u013a\7\"\2\2\u013a\u0154\3\2\2\2\u013b\u013c\5*\26\2\u013c"+
		"\u013d\7\35\2\2\u013d\u013e\5*\26\2\u013e\u0154\3\2\2\2\u013f\u0140\5"+
		"*\26\2\u0140\u0141\7\37\2\2\u0141\u0142\5*\26\2\u0142\u0154\3\2\2\2\u0143"+
		"\u0144\5*\26\2\u0144\u0145\7\36\2\2\u0145\u0146\5*\26\2\u0146\u0154\3"+
		"\2\2\2\u0147\u0148\5*\26\2\u0148\u0149\7\31\2\2\u0149\u014a\5*\26\2\u014a"+
		"\u0154\3\2\2\2\u014b\u014c\5*\26\2\u014c\u014d\7\33\2\2\u014d\u014e\5"+
		"*\26\2\u014e\u0154\3\2\2\2\u014f\u0150\5*\26\2\u0150\u0151\7\6\2\2\u0151"+
		"\u0152\5*\26\2\u0152\u0154\3\2\2\2\u0153\u0136\3\2\2\2\u0153\u013b\3\2"+
		"\2\2\u0153\u013f\3\2\2\2\u0153\u0143\3\2\2\2\u0153\u0147\3\2\2\2\u0153"+
		"\u014b\3\2\2\2\u0153\u014f\3\2\2\2\u0154\u015d\3\2\2\2\u0155\u0156\f\5"+
		"\2\2\u0156\u0157\7\27\2\2\u0157\u015c\5,\27\6\u0158\u0159\f\4\2\2\u0159"+
		"\u015a\7\30\2\2\u015a\u015c\5,\27\5\u015b\u0155\3\2\2\2\u015b\u0158\3"+
		"\2\2\2\u015c\u015f\3\2\2\2\u015d\u015b\3\2\2\2\u015d\u015e\3\2\2\2\u015e"+
		"-\3\2\2\2\u015f\u015d\3\2\2\2\23OUi\u0084\u008d\u0093\u009e\u00a5\u00d5"+
		"\u0108\u010a\u0120\u0131\u0133\u0153\u015b\u015d";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}