// Generated from /home/avaljot/cf/constraintflow/python/dsl.g4 by ANTLR 4.9.2
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class dslLexer extends Lexer {
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
	public static String[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN"
	};

	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	private static String[] makeRuleNames() {
		return new String[] {
			"FLOW", "ARROW", "TRANSFORMER", "IN", "OUT", "BACKWARD", "FORWARD", "INTT", 
			"FLOATT", "BOOL", "POLYEXP", "ZONOEXP", "NEURON", "LIST", "DOT", "COMMA", 
			"PLUS", "MINUS", "MULT", "DIV", "AND", "OR", "LT", "EQ", "EQQ", "NEQ", 
			"GT", "LEQ", "GEQ", "NOT", "LPAREN", "RPAREN", "LSQR", "RSQR", "LBRACE", 
			"RBRACE", "SEMI", "QUES", "COLON", "IF", "TRAV", "SUM", "LEN", "AVG", 
			"SUB", "MAP", "DOTT", "ARGMIN", "ARGMAX", "MIN", "MAX", "WEIGHT", "BIAS", 
			"LAYER", "AFFINE", "RELU", "MAXPOOL", "SIGMOID", "TANH", "SHAPE", "FUNC", 
			"EPSILON", "TRUE", "FALSE", "CURR", "PREV", "IntConst", "FloatConst", 
			"Digit", "Sign", "VAR", "Nondigit", "WS", "LineComment"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'flow'", "'->'", "'transformer'", "'in'", "'out'", "'backward'", 
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


	public dslLexer(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "dsl.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getChannelNames() { return channelNames; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2I\u01fe\b\1\4\2\t"+
		"\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13"+
		"\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22"+
		"\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31\t\31"+
		"\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36\t\36\4\37\t\37\4 \t \4!"+
		"\t!\4\"\t\"\4#\t#\4$\t$\4%\t%\4&\t&\4\'\t\'\4(\t(\4)\t)\4*\t*\4+\t+\4"+
		",\t,\4-\t-\4.\t.\4/\t/\4\60\t\60\4\61\t\61\4\62\t\62\4\63\t\63\4\64\t"+
		"\64\4\65\t\65\4\66\t\66\4\67\t\67\48\t8\49\t9\4:\t:\4;\t;\4<\t<\4=\t="+
		"\4>\t>\4?\t?\4@\t@\4A\tA\4B\tB\4C\tC\4D\tD\4E\tE\4F\tF\4G\tG\4H\tH\4I"+
		"\tI\4J\tJ\4K\tK\3\2\3\2\3\2\3\2\3\2\3\3\3\3\3\3\3\4\3\4\3\4\3\4\3\4\3"+
		"\4\3\4\3\4\3\4\3\4\3\4\3\4\3\5\3\5\3\5\3\6\3\6\3\6\3\6\3\7\3\7\3\7\3\7"+
		"\3\7\3\7\3\7\3\7\3\7\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\t\3\t\3\t\3\t\3"+
		"\n\3\n\3\n\3\n\3\n\3\n\3\13\3\13\3\13\3\13\3\13\3\f\3\f\3\f\3\f\3\f\3"+
		"\f\3\f\3\f\3\r\3\r\3\r\3\r\3\r\3\r\3\r\3\r\3\16\3\16\3\16\3\16\3\16\3"+
		"\16\3\16\3\17\3\17\3\17\3\17\3\17\3\20\3\20\3\21\3\21\3\22\3\22\3\23\3"+
		"\23\3\24\3\24\3\25\3\25\3\26\3\26\3\26\3\26\3\27\3\27\3\27\3\30\3\30\3"+
		"\31\3\31\3\32\3\32\3\32\3\33\3\33\3\33\3\34\3\34\3\35\3\35\3\35\3\36\3"+
		"\36\3\36\3\37\3\37\3 \3 \3!\3!\3\"\3\"\3#\3#\3$\3$\3%\3%\3&\3&\3\'\3\'"+
		"\3(\3(\3)\3)\3)\3*\3*\3*\3*\3*\3*\3*\3*\3*\3+\3+\3+\3+\3,\3,\3,\3,\3-"+
		"\3-\3-\3-\3.\3.\3.\3.\3/\3/\3/\3/\3\60\3\60\3\60\3\60\3\61\3\61\3\61\3"+
		"\61\3\61\3\61\3\61\3\62\3\62\3\62\3\62\3\62\3\62\3\62\3\63\3\63\3\63\3"+
		"\63\3\64\3\64\3\64\3\64\3\65\3\65\3\65\3\65\3\65\3\65\3\65\3\66\3\66\3"+
		"\66\3\66\3\66\3\67\3\67\3\67\3\67\3\67\3\67\38\38\38\38\38\38\38\39\3"+
		"9\39\39\39\3:\3:\3:\3:\3:\3:\3:\3:\3;\3;\3;\3;\3;\3;\3;\3;\3<\3<\3<\3"+
		"<\3<\3=\3=\3=\3=\3=\3=\3=\3=\3=\3=\3=\3=\3=\3>\3>\3>\3>\3>\3?\3?\3?\3"+
		"?\3@\3@\3@\3@\3@\3A\3A\3A\3A\3A\3A\3B\3B\3B\3B\3B\3C\3C\3C\3C\3C\3D\5"+
		"D\u01c1\nD\3D\6D\u01c4\nD\rD\16D\u01c5\3E\6E\u01c9\nE\rE\16E\u01ca\3E"+
		"\3E\6E\u01cf\nE\rE\16E\u01d0\3E\3E\5E\u01d5\nE\3E\6E\u01d8\nE\rE\16E\u01d9"+
		"\5E\u01dc\nE\3F\3F\3G\3G\3H\3H\3H\3H\7H\u01e6\nH\fH\16H\u01e9\13H\3I\3"+
		"I\3J\6J\u01ee\nJ\rJ\16J\u01ef\3J\3J\3K\3K\3K\3K\7K\u01f8\nK\fK\16K\u01fb"+
		"\13K\3K\3K\2\2L\3\3\5\4\7\5\t\6\13\7\r\b\17\t\21\n\23\13\25\f\27\r\31"+
		"\16\33\17\35\20\37\21!\22#\23%\24\'\25)\26+\27-\30/\31\61\32\63\33\65"+
		"\34\67\359\36;\37= ?!A\"C#E$G%I&K\'M(O)Q*S+U,W-Y.[/]\60_\61a\62c\63e\64"+
		"g\65i\66k\67m8o9q:s;u<w=y>{?}@\177A\u0081B\u0083C\u0085D\u0087E\u0089"+
		"F\u008b\2\u008d\2\u008fG\u0091\2\u0093H\u0095I\3\2\b\3\2\62;\4\2GGgg\4"+
		"\2--//\5\2C\\aac|\5\2\13\f\17\17\"\"\4\2\f\f\17\17\2\u0206\2\3\3\2\2\2"+
		"\2\5\3\2\2\2\2\7\3\2\2\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2"+
		"\2\2\2\21\3\2\2\2\2\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2\31\3\2\2\2"+
		"\2\33\3\2\2\2\2\35\3\2\2\2\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2\2\2%\3\2\2"+
		"\2\2\'\3\2\2\2\2)\3\2\2\2\2+\3\2\2\2\2-\3\2\2\2\2/\3\2\2\2\2\61\3\2\2"+
		"\2\2\63\3\2\2\2\2\65\3\2\2\2\2\67\3\2\2\2\29\3\2\2\2\2;\3\2\2\2\2=\3\2"+
		"\2\2\2?\3\2\2\2\2A\3\2\2\2\2C\3\2\2\2\2E\3\2\2\2\2G\3\2\2\2\2I\3\2\2\2"+
		"\2K\3\2\2\2\2M\3\2\2\2\2O\3\2\2\2\2Q\3\2\2\2\2S\3\2\2\2\2U\3\2\2\2\2W"+
		"\3\2\2\2\2Y\3\2\2\2\2[\3\2\2\2\2]\3\2\2\2\2_\3\2\2\2\2a\3\2\2\2\2c\3\2"+
		"\2\2\2e\3\2\2\2\2g\3\2\2\2\2i\3\2\2\2\2k\3\2\2\2\2m\3\2\2\2\2o\3\2\2\2"+
		"\2q\3\2\2\2\2s\3\2\2\2\2u\3\2\2\2\2w\3\2\2\2\2y\3\2\2\2\2{\3\2\2\2\2}"+
		"\3\2\2\2\2\177\3\2\2\2\2\u0081\3\2\2\2\2\u0083\3\2\2\2\2\u0085\3\2\2\2"+
		"\2\u0087\3\2\2\2\2\u0089\3\2\2\2\2\u008f\3\2\2\2\2\u0093\3\2\2\2\2\u0095"+
		"\3\2\2\2\3\u0097\3\2\2\2\5\u009c\3\2\2\2\7\u009f\3\2\2\2\t\u00ab\3\2\2"+
		"\2\13\u00ae\3\2\2\2\r\u00b2\3\2\2\2\17\u00bb\3\2\2\2\21\u00c3\3\2\2\2"+
		"\23\u00c7\3\2\2\2\25\u00cd\3\2\2\2\27\u00d2\3\2\2\2\31\u00da\3\2\2\2\33"+
		"\u00e2\3\2\2\2\35\u00e9\3\2\2\2\37\u00ee\3\2\2\2!\u00f0\3\2\2\2#\u00f2"+
		"\3\2\2\2%\u00f4\3\2\2\2\'\u00f6\3\2\2\2)\u00f8\3\2\2\2+\u00fa\3\2\2\2"+
		"-\u00fe\3\2\2\2/\u0101\3\2\2\2\61\u0103\3\2\2\2\63\u0105\3\2\2\2\65\u0108"+
		"\3\2\2\2\67\u010b\3\2\2\29\u010d\3\2\2\2;\u0110\3\2\2\2=\u0113\3\2\2\2"+
		"?\u0115\3\2\2\2A\u0117\3\2\2\2C\u0119\3\2\2\2E\u011b\3\2\2\2G\u011d\3"+
		"\2\2\2I\u011f\3\2\2\2K\u0121\3\2\2\2M\u0123\3\2\2\2O\u0125\3\2\2\2Q\u0127"+
		"\3\2\2\2S\u012a\3\2\2\2U\u0133\3\2\2\2W\u0137\3\2\2\2Y\u013b\3\2\2\2["+
		"\u013f\3\2\2\2]\u0143\3\2\2\2_\u0147\3\2\2\2a\u014b\3\2\2\2c\u0152\3\2"+
		"\2\2e\u0159\3\2\2\2g\u015d\3\2\2\2i\u0161\3\2\2\2k\u0168\3\2\2\2m\u016d"+
		"\3\2\2\2o\u0173\3\2\2\2q\u017a\3\2\2\2s\u017f\3\2\2\2u\u0187\3\2\2\2w"+
		"\u018f\3\2\2\2y\u0194\3\2\2\2{\u01a1\3\2\2\2}\u01a6\3\2\2\2\177\u01aa"+
		"\3\2\2\2\u0081\u01af\3\2\2\2\u0083\u01b5\3\2\2\2\u0085\u01ba\3\2\2\2\u0087"+
		"\u01c0\3\2\2\2\u0089\u01c8\3\2\2\2\u008b\u01dd\3\2\2\2\u008d\u01df\3\2"+
		"\2\2\u008f\u01e1\3\2\2\2\u0091\u01ea\3\2\2\2\u0093\u01ed\3\2\2\2\u0095"+
		"\u01f3\3\2\2\2\u0097\u0098\7h\2\2\u0098\u0099\7n\2\2\u0099\u009a\7q\2"+
		"\2\u009a\u009b\7y\2\2\u009b\4\3\2\2\2\u009c\u009d\7/\2\2\u009d\u009e\7"+
		"@\2\2\u009e\6\3\2\2\2\u009f\u00a0\7v\2\2\u00a0\u00a1\7t\2\2\u00a1\u00a2"+
		"\7c\2\2\u00a2\u00a3\7p\2\2\u00a3\u00a4\7u\2\2\u00a4\u00a5\7h\2\2\u00a5"+
		"\u00a6\7q\2\2\u00a6\u00a7\7t\2\2\u00a7\u00a8\7o\2\2\u00a8\u00a9\7g\2\2"+
		"\u00a9\u00aa\7t\2\2\u00aa\b\3\2\2\2\u00ab\u00ac\7k\2\2\u00ac\u00ad\7p"+
		"\2\2\u00ad\n\3\2\2\2\u00ae\u00af\7q\2\2\u00af\u00b0\7w\2\2\u00b0\u00b1"+
		"\7v\2\2\u00b1\f\3\2\2\2\u00b2\u00b3\7d\2\2\u00b3\u00b4\7c\2\2\u00b4\u00b5"+
		"\7e\2\2\u00b5\u00b6\7m\2\2\u00b6\u00b7\7y\2\2\u00b7\u00b8\7c\2\2\u00b8"+
		"\u00b9\7t\2\2\u00b9\u00ba\7f\2\2\u00ba\16\3\2\2\2\u00bb\u00bc\7h\2\2\u00bc"+
		"\u00bd\7q\2\2\u00bd\u00be\7t\2\2\u00be\u00bf\7y\2\2\u00bf\u00c0\7c\2\2"+
		"\u00c0\u00c1\7t\2\2\u00c1\u00c2\7f\2\2\u00c2\20\3\2\2\2\u00c3\u00c4\7"+
		"K\2\2\u00c4\u00c5\7p\2\2\u00c5\u00c6\7v\2\2\u00c6\22\3\2\2\2\u00c7\u00c8"+
		"\7H\2\2\u00c8\u00c9\7n\2\2\u00c9\u00ca\7q\2\2\u00ca\u00cb\7c\2\2\u00cb"+
		"\u00cc\7v\2\2\u00cc\24\3\2\2\2\u00cd\u00ce\7D\2\2\u00ce\u00cf\7q\2\2\u00cf"+
		"\u00d0\7q\2\2\u00d0\u00d1\7n\2\2\u00d1\26\3\2\2\2\u00d2\u00d3\7R\2\2\u00d3"+
		"\u00d4\7q\2\2\u00d4\u00d5\7n\2\2\u00d5\u00d6\7{\2\2\u00d6\u00d7\7G\2\2"+
		"\u00d7\u00d8\7z\2\2\u00d8\u00d9\7r\2\2\u00d9\30\3\2\2\2\u00da\u00db\7"+
		"\\\2\2\u00db\u00dc\7q\2\2\u00dc\u00dd\7p\2\2\u00dd\u00de\7q\2\2\u00de"+
		"\u00df\7G\2\2\u00df\u00e0\7z\2\2\u00e0\u00e1\7r\2\2\u00e1\32\3\2\2\2\u00e2"+
		"\u00e3\7P\2\2\u00e3\u00e4\7g\2\2\u00e4\u00e5\7w\2\2\u00e5\u00e6\7t\2\2"+
		"\u00e6\u00e7\7q\2\2\u00e7\u00e8\7p\2\2\u00e8\34\3\2\2\2\u00e9\u00ea\7"+
		"N\2\2\u00ea\u00eb\7k\2\2\u00eb\u00ec\7u\2\2\u00ec\u00ed\7v\2\2\u00ed\36"+
		"\3\2\2\2\u00ee\u00ef\7\60\2\2\u00ef \3\2\2\2\u00f0\u00f1\7.\2\2\u00f1"+
		"\"\3\2\2\2\u00f2\u00f3\7-\2\2\u00f3$\3\2\2\2\u00f4\u00f5\7/\2\2\u00f5"+
		"&\3\2\2\2\u00f6\u00f7\7,\2\2\u00f7(\3\2\2\2\u00f8\u00f9\7\61\2\2\u00f9"+
		"*\3\2\2\2\u00fa\u00fb\7c\2\2\u00fb\u00fc\7p\2\2\u00fc\u00fd\7f\2\2\u00fd"+
		",\3\2\2\2\u00fe\u00ff\7q\2\2\u00ff\u0100\7t\2\2\u0100.\3\2\2\2\u0101\u0102"+
		"\7>\2\2\u0102\60\3\2\2\2\u0103\u0104\7?\2\2\u0104\62\3\2\2\2\u0105\u0106"+
		"\7?\2\2\u0106\u0107\7?\2\2\u0107\64\3\2\2\2\u0108\u0109\7#\2\2\u0109\u010a"+
		"\7?\2\2\u010a\66\3\2\2\2\u010b\u010c\7@\2\2\u010c8\3\2\2\2\u010d\u010e"+
		"\7>\2\2\u010e\u010f\7?\2\2\u010f:\3\2\2\2\u0110\u0111\7@\2\2\u0111\u0112"+
		"\7?\2\2\u0112<\3\2\2\2\u0113\u0114\7#\2\2\u0114>\3\2\2\2\u0115\u0116\7"+
		"*\2\2\u0116@\3\2\2\2\u0117\u0118\7+\2\2\u0118B\3\2\2\2\u0119\u011a\7]"+
		"\2\2\u011aD\3\2\2\2\u011b\u011c\7_\2\2\u011cF\3\2\2\2\u011d\u011e\7}\2"+
		"\2\u011eH\3\2\2\2\u011f\u0120\7\177\2\2\u0120J\3\2\2\2\u0121\u0122\7="+
		"\2\2\u0122L\3\2\2\2\u0123\u0124\7A\2\2\u0124N\3\2\2\2\u0125\u0126\7<\2"+
		"\2\u0126P\3\2\2\2\u0127\u0128\7k\2\2\u0128\u0129\7h\2\2\u0129R\3\2\2\2"+
		"\u012a\u012b\7v\2\2\u012b\u012c\7t\2\2\u012c\u012d\7c\2\2\u012d\u012e"+
		"\7x\2\2\u012e\u012f\7g\2\2\u012f\u0130\7t\2\2\u0130\u0131\7u\2\2\u0131"+
		"\u0132\7g\2\2\u0132T\3\2\2\2\u0133\u0134\7u\2\2\u0134\u0135\7w\2\2\u0135"+
		"\u0136\7o\2\2\u0136V\3\2\2\2\u0137\u0138\7n\2\2\u0138\u0139\7g\2\2\u0139"+
		"\u013a\7p\2\2\u013aX\3\2\2\2\u013b\u013c\7c\2\2\u013c\u013d\7x\2\2\u013d"+
		"\u013e\7i\2\2\u013eZ\3\2\2\2\u013f\u0140\7u\2\2\u0140\u0141\7w\2\2\u0141"+
		"\u0142\7d\2\2\u0142\\\3\2\2\2\u0143\u0144\7o\2\2\u0144\u0145\7c\2\2\u0145"+
		"\u0146\7r\2\2\u0146^\3\2\2\2\u0147\u0148\7f\2\2\u0148\u0149\7q\2\2\u0149"+
		"\u014a\7v\2\2\u014a`\3\2\2\2\u014b\u014c\7c\2\2\u014c\u014d\7t\2\2\u014d"+
		"\u014e\7i\2\2\u014e\u014f\7o\2\2\u014f\u0150\7k\2\2\u0150\u0151\7p\2\2"+
		"\u0151b\3\2\2\2\u0152\u0153\7c\2\2\u0153\u0154\7t\2\2\u0154\u0155\7i\2"+
		"\2\u0155\u0156\7o\2\2\u0156\u0157\7c\2\2\u0157\u0158\7z\2\2\u0158d\3\2"+
		"\2\2\u0159\u015a\7o\2\2\u015a\u015b\7k\2\2\u015b\u015c\7p\2\2\u015cf\3"+
		"\2\2\2\u015d\u015e\7o\2\2\u015e\u015f\7c\2\2\u015f\u0160\7z\2\2\u0160"+
		"h\3\2\2\2\u0161\u0162\7y\2\2\u0162\u0163\7g\2\2\u0163\u0164\7k\2\2\u0164"+
		"\u0165\7i\2\2\u0165\u0166\7j\2\2\u0166\u0167\7v\2\2\u0167j\3\2\2\2\u0168"+
		"\u0169\7d\2\2\u0169\u016a\7k\2\2\u016a\u016b\7c\2\2\u016b\u016c\7u\2\2"+
		"\u016cl\3\2\2\2\u016d\u016e\7n\2\2\u016e\u016f\7c\2\2\u016f\u0170\7{\2"+
		"\2\u0170\u0171\7g\2\2\u0171\u0172\7t\2\2\u0172n\3\2\2\2\u0173\u0174\7"+
		"C\2\2\u0174\u0175\7h\2\2\u0175\u0176\7h\2\2\u0176\u0177\7k\2\2\u0177\u0178"+
		"\7p\2\2\u0178\u0179\7g\2\2\u0179p\3\2\2\2\u017a\u017b\7T\2\2\u017b\u017c"+
		"\7g\2\2\u017c\u017d\7n\2\2\u017d\u017e\7w\2\2\u017er\3\2\2\2\u017f\u0180"+
		"\7O\2\2\u0180\u0181\7c\2\2\u0181\u0182\7z\2\2\u0182\u0183\7r\2\2\u0183"+
		"\u0184\7q\2\2\u0184\u0185\7q\2\2\u0185\u0186\7n\2\2\u0186t\3\2\2\2\u0187"+
		"\u0188\7U\2\2\u0188\u0189\7k\2\2\u0189\u018a\7i\2\2\u018a\u018b\7o\2\2"+
		"\u018b\u018c\7q\2\2\u018c\u018d\7k\2\2\u018d\u018e\7f\2\2\u018ev\3\2\2"+
		"\2\u018f\u0190\7V\2\2\u0190\u0191\7c\2\2\u0191\u0192\7p\2\2\u0192\u0193"+
		"\7j\2\2\u0193x\3\2\2\2\u0194\u0195\7f\2\2\u0195\u0196\7g\2\2\u0196\u0197"+
		"\7h\2\2\u0197\u0198\7\"\2\2\u0198\u0199\7U\2\2\u0199\u019a\7j\2\2\u019a"+
		"\u019b\7c\2\2\u019b\u019c\7r\2\2\u019c\u019d\7g\2\2\u019d\u019e\7\"\2"+
		"\2\u019e\u019f\7c\2\2\u019f\u01a0\7u\2\2\u01a0z\3\2\2\2\u01a1\u01a2\7"+
		"h\2\2\u01a2\u01a3\7w\2\2\u01a3\u01a4\7p\2\2\u01a4\u01a5\7e\2\2\u01a5|"+
		"\3\2\2\2\u01a6\u01a7\7g\2\2\u01a7\u01a8\7r\2\2\u01a8\u01a9\7u\2\2\u01a9"+
		"~\3\2\2\2\u01aa\u01ab\7v\2\2\u01ab\u01ac\7t\2\2\u01ac\u01ad\7w\2\2\u01ad"+
		"\u01ae\7g\2\2\u01ae\u0080\3\2\2\2\u01af\u01b0\7h\2\2\u01b0\u01b1\7c\2"+
		"\2\u01b1\u01b2\7n\2\2\u01b2\u01b3\7u\2\2\u01b3\u01b4\7g\2\2\u01b4\u0082"+
		"\3\2\2\2\u01b5\u01b6\7e\2\2\u01b6\u01b7\7w\2\2\u01b7\u01b8\7t\2\2\u01b8"+
		"\u01b9\7t\2\2\u01b9\u0084\3\2\2\2\u01ba\u01bb\7r\2\2\u01bb\u01bc\7t\2"+
		"\2\u01bc\u01bd\7g\2\2\u01bd\u01be\7x\2\2\u01be\u0086\3\2\2\2\u01bf\u01c1"+
		"\5\u008dG\2\u01c0\u01bf\3\2\2\2\u01c0\u01c1\3\2\2\2\u01c1\u01c3\3\2\2"+
		"\2\u01c2\u01c4\5\u008bF\2\u01c3\u01c2\3\2\2\2\u01c4\u01c5\3\2\2\2\u01c5"+
		"\u01c3\3\2\2\2\u01c5\u01c6\3\2\2\2\u01c6\u0088\3\2\2\2\u01c7\u01c9\t\2"+
		"\2\2\u01c8\u01c7\3\2\2\2\u01c9\u01ca\3\2\2\2\u01ca\u01c8\3\2\2\2\u01ca"+
		"\u01cb\3\2\2\2\u01cb\u01cc\3\2\2\2\u01cc\u01ce\7\60\2\2\u01cd\u01cf\t"+
		"\2\2\2\u01ce\u01cd\3\2\2\2\u01cf\u01d0\3\2\2\2\u01d0\u01ce\3\2\2\2\u01d0"+
		"\u01d1\3\2\2\2\u01d1\u01db\3\2\2\2\u01d2\u01d4\t\3\2\2\u01d3\u01d5\t\4"+
		"\2\2\u01d4\u01d3\3\2\2\2\u01d4\u01d5\3\2\2\2\u01d5\u01d7\3\2\2\2\u01d6"+
		"\u01d8\t\2\2\2\u01d7\u01d6\3\2\2\2\u01d8\u01d9\3\2\2\2\u01d9\u01d7\3\2"+
		"\2\2\u01d9\u01da\3\2\2\2\u01da\u01dc\3\2\2\2\u01db\u01d2\3\2\2\2\u01db"+
		"\u01dc\3\2\2\2\u01dc\u008a\3\2\2\2\u01dd\u01de\t\2\2\2\u01de\u008c\3\2"+
		"\2\2\u01df\u01e0\t\4\2\2\u01e0\u008e\3\2\2\2\u01e1\u01e7\5\u0091I\2\u01e2"+
		"\u01e6\5\u0091I\2\u01e3\u01e6\5\u008bF\2\u01e4\u01e6\7)\2\2\u01e5\u01e2"+
		"\3\2\2\2\u01e5\u01e3\3\2\2\2\u01e5\u01e4\3\2\2\2\u01e6\u01e9\3\2\2\2\u01e7"+
		"\u01e5\3\2\2\2\u01e7\u01e8\3\2\2\2\u01e8\u0090\3\2\2\2\u01e9\u01e7\3\2"+
		"\2\2\u01ea\u01eb\t\5\2\2\u01eb\u0092\3\2\2\2\u01ec\u01ee\t\6\2\2\u01ed"+
		"\u01ec\3\2\2\2\u01ee\u01ef\3\2\2\2\u01ef\u01ed\3\2\2\2\u01ef\u01f0\3\2"+
		"\2\2\u01f0\u01f1\3\2\2\2\u01f1\u01f2\bJ\2\2\u01f2\u0094\3\2\2\2\u01f3"+
		"\u01f4\7\61\2\2\u01f4\u01f5\7\61\2\2\u01f5\u01f9\3\2\2\2\u01f6\u01f8\n"+
		"\7\2\2\u01f7\u01f6\3\2\2\2\u01f8\u01fb\3\2\2\2\u01f9\u01f7\3\2\2\2\u01f9"+
		"\u01fa\3\2\2\2\u01fa\u01fc\3\2\2\2\u01fb\u01f9\3\2\2\2\u01fc\u01fd\bK"+
		"\3\2\u01fd\u0096\3\2\2\2\16\2\u01c0\u01c5\u01ca\u01d0\u01d4\u01d9\u01db"+
		"\u01e5\u01e7\u01ef\u01f9\4\b\2\2\2\3\2";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}