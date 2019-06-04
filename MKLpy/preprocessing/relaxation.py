


C_UPPER   = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
C_LOWER   = tuple('abcdefghijklmnopqrstuvwxyz')
C_NUMBER  = tuple('1234567890')
C_SYMBOLS = tuple('!"£$%&/()=?^§<>.:,;-_+*[]#')






def _get_map(relaxation):
	relaxation = relaxation or {'A':C_UPPER, 'a':C_LOWER, '0':C_NUMBER, '+':C_SYMBOLS}
	relaxation = {str(c): group for c, group in enumerate(relaxation)} if type(relaxation) != dict else relaxation
	return {c: k for k,group in relaxation.items() for c in group}, '-'



def sequences_relaxation(X, relaxation=None):
	'''
	Lhis function applies a transformation to the input strings.
	relaxation is a dict with form
	{c1 : list_of_chars, ...}
	the function substitutes the characters in list_of_chars with the symbol c
	If relaxation is not a dict, but a sequence-like of list(s)_of_chats, then the symbols c are automatically inferred.
	'''
	relaxation, unknown = _get_map(relaxation)
	return [''.join(relaxation[c] if c in relaxation else unknown for c in x) for x in X]


def sequence_relaxation(X, relaxation=None):
	relaxation, unknown = _get_map(relaxation)
	return ''.join(relaxation[c] if c in relaxation else unknown for c in x)