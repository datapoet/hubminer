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
package learning.supervised.methods.discrete;

import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import java.io.Serializable;
import java.util.HashMap;
import learning.supervised.DiscreteCategory;
import learning.supervised.DiscreteClassifier;
import learning.supervised.evaluation.ValidateableInterface;

/**
 * This class implements the trivial zero-rule classifier.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DZeroRule extends DiscreteClassifier implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private int majorityClassIndex = 0;
    private float[] classDistributions = null;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        return paramMap;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "Zero-Rule";
    }

    /**
     * The default constructor.
     */
    public DZeroRule() {
    }

    /**
     * Initialization.
     *
     * @param discDSet DiscretizedDataSet that is the training data.
     */
    public DZeroRule(DiscretizedDataSet discDSet) {
        setDataType(discDSet);
        generateClassesFromDataType();
    }

    /**
     * Initialization
     *
     * @param discDSet DiscretizedDataSet that is the data context.
     * @param dataClasses DiscreteCategory[] that is the training data.
     */
    public DZeroRule(DiscretizedDataSet discDSet,
            DiscreteCategory[] dataClasses) {
        setDataType(discDSet);
        setClasses(dataClasses);
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        return new DZeroRule();
    }

    @Override
    public void train() throws Exception {
        DiscreteCategory[] dataClasses = getClasses();
        DiscretizedDataSet discDSet = getDataType();
        int maxClassSize = 0;
        classDistributions = new float[dataClasses.length];
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            classDistributions[cIndex] = dataClasses[cIndex].indexes.size()
                    / (float) discDSet.data.size();
            if (dataClasses[cIndex].indexes.size() > maxClassSize) {
                maxClassSize = dataClasses[cIndex].indexes.size();
                majorityClassIndex = cIndex;
            }
        }
    }

    @Override
    public int classify(DiscretizedDataInstance instance) throws Exception {
        return majorityClassIndex;
    }

    @Override
    public float[] classifyProbabilistically(DiscretizedDataInstance instance)
            throws Exception {
        return classDistributions;
    }
}
