import pytest
from dynamic_sliding_window import DynamicSlidingWindow
'''
Test suite to test the functionality of DynamicSlidingWindow inference regime
'''

@pytest.mark.parameterize("lang", ["eng", "jpn", "fra"])
def test_unsupported_languages(lang):
    '''
    test that unsupported languages raise error
    the only supported languages are cmn, deu and ita.
    '''
    #create an instance of the sliding window class without loading the model
    sliding_window = DynamicSlidingWindow.__new__(DynamicSlidingWindow) 
    with pytest.raises(ValueError, match="Unsupported language"):
        sliding_window.set_target_language(lang)



