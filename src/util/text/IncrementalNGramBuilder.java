/**
* Hub Miner: a hubness-aware machine learning experimentation library.
* Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
* 
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*/
package util.text;

import data.representation.sparse.BOWDataSet;
import data.representation.sparse.BOWInstance;
import java.util.ArrayList;

/**
 * Incrementally builds an ngram representation from text input.
 *
 * @author Nenad
 */
public class IncrementalNGramBuilder {

    private BOWDataSet ngramDSet;
    private int nGramSize;

    /**
     * @return BOWDataSet object containing the n-gram representation.
     */
    public BOWDataSet getNGramDataSet() {
        return ngramDSet;
    }

    /**
     * Initialize the fields.
     *
     * @param nGramSize Integer that is the length of ngrams.
     * @param initialVocabularySize Integer that is the initial vocabulary size.
     */
    public IncrementalNGramBuilder(int nGramSize, int initialVocabularySize) {
        ngramDSet = new BOWDataSet(initialVocabularySize);
        ngramDSet.data = new ArrayList<>(2000);
        this.nGramSize = nGramSize;
    }

    /**
     * Adds a new DataInstance from the provided text and updates the current
     * representation.
     *
     * @param text String that is to be parsed into a DataInstance.
     */
    public void addInstanceFromString(String text) {
        if (text == null || text.length() == 0) {
            ngramDSet.addDataInstance(new BOWInstance(ngramDSet));
            return;
        }
        BOWInstance instance = new BOWInstance(ngramDSet);
        StringBuilder ngram = new StringBuilder();
        for (int i = 0; i < Math.min(nGramSize, text.length()); i++) {
            ngram.append(text.charAt(i));
        }
        for (int i = 0; i < (nGramSize - text.length()); i++) {
            ngram.append(' ');
        }
        instance.addWord(ngram.toString());
        for (int i = 0; i < text.length() - nGramSize - 1; i++) {
            for (int j = 0; j < nGramSize - 1; j++) {
                ngram.setCharAt(j, text.charAt(i + j + 1));
            }
            ngram.setCharAt(nGramSize - 1, text.charAt(i + nGramSize));
            instance.addWord(ngram.toString());
        }
        ngramDSet.addDataInstance(instance);
    }

    /**
     * Adds a new DataInstance from the provided text and updates the current
     * representation.
     *
     * @param text String that is to be parsed into a DataInstance.
     * @param label Integer that is the instance label.
     */
    public void addInstanceFromString(String text, int label) {
        if (text == null || text.length() == 0) {
            ngramDSet.addDataInstance(new BOWInstance(ngramDSet));
            return;
        }
        BOWInstance instance = new BOWInstance(ngramDSet);
        instance.setCategory(label);
        StringBuilder ngram = new StringBuilder();
        for (int i = 0; i < Math.min(nGramSize, text.length()); i++) {
            ngram.append(text.charAt(i));
        }
        for (int i = 0; i < (nGramSize - text.length()); i++) {
            ngram.append(' ');
        }
        instance.addWord(ngram.toString());
        for (int i = 0; i < text.length() - nGramSize - 1; i++) {
            for (int j = 0; j < nGramSize - 1; j++) {
                ngram.setCharAt(j, text.charAt(i + j + 1));
            }
            ngram.setCharAt(nGramSize - 1, text.charAt(i + nGramSize));
            instance.addWord(ngram.toString());
        }
        ngramDSet.addDataInstance(instance);
    }
}
