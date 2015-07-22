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
package feature.evaluation;

import data.representation.discrete.DiscretizedDataSet;
import java.util.ArrayList;

/**
 * This class and its implementations deal with assessing the quality of a split
 * on an attribute in tree-like classification scenarios.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class SplitAssesment {

    /**
     * Initialization.
     *
     * @param dset DataSet to analyze.
     */
    public SplitAssesment(DiscretizedDataSet dset) {
        this.dset = dset;
    }
    private DiscretizedDataSet dset;

    /**
     * Sets the dataset to analyze.
     *
     * @param dset DataSet object to analyze.
     */
    public void setDataContext(DiscretizedDataSet dset) {
        this.dset = dset;
    }

    /**
     * @return The DataSet object that is being analyzed.
     */
    public DiscretizedDataSet getDataContext() {
        return dset;
    }

    /**
     * Asses the quality of the attribute split on the whole dataset.
     *
     * @param split An array of ArrayList-s of integer indexes of DataInstance
     * objects belonging to different parts of the feature split.
     * @return The quality assessment of the split.
     */
    public abstract float assesSplitOnWhole(ArrayList<Integer>[] split);

    /**
     * Asses the quality of the attribute split on a subset of the data.
     *
     * @param subset An ArrayList of integer indexes of DataInstance objects to
     * assess the split on.
     * @param split An array of ArrayList-s of integer indexes of DataInstance
     * objects belonging to different parts of the feature split.
     * @return The quality assessment of the split.
     */
    public abstract float assesSplitOnSubset(ArrayList<Integer> subset,
            ArrayList<Integer>[] split);
}
