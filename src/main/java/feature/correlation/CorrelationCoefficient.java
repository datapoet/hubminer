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
package feature.correlation;

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;

/**
 * This class sets up the basics of correlation testing in datasets and is meant
 * to be extended to specific correlation measures.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CorrelationCoefficient {

    /**
     * Tests for correlation.
     *
     * @param dset DataSet data context, if needed.
     * @param first First value array.
     * @param second Second value array.
     * @return Correlation measure.
     * @throws Exception
     */
    public static float correlation(DataSet dset, float[] first, float[] second)
            throws Exception {
        return 0;
    }

    /**
     * Tests for correlation between different features of a same dataset.
     *
     * @param dset DataSet data context.
     * @param type Feature type, as in DataMineConstants.
     * @param indexFirst Index of the first feature to compare.
     * @param indexSecond Index of the second feature to compare.
     * @return Correlation measure between the two features, as a float value.
     * @throws Exception
     */
    public static float correlation(DataSet dset, int type, int indexFirst,
            int indexSecond) throws Exception {
        if (dset == null || dset.isEmpty()) {
            return 0;
        }
        if (type == DataMineConstants.FLOAT) {
            // Range check.
            if (indexFirst >= dset.fAttrNames.length
                    || indexSecond >= dset.fAttrNames.length) {
                return 0;
            } else {
                float[] first = new float[dset.size()];
                float[] second = new float[dset.size()];
                DataInstance instance;
                for (int i = 0; i < dset.size(); i++) {
                    instance = dset.data.get(i);
                    first[i] = instance.fAttr[indexFirst];
                    second[i] = instance.fAttr[indexSecond];
                }
                return correlation(dset, first, second);
            }
        } else {
            // Range check.
            if (indexFirst >= dset.iAttrNames.length
                    || indexSecond >= dset.iAttrNames.length) {
                return 0;
            } else {
                float[] first = new float[dset.size()];
                float[] second = new float[dset.size()];
                DataInstance instance;
                for (int i = 0; i < dset.size(); i++) {
                    instance = dset.data.get(i);
                    first[i] = instance.iAttr[indexFirst];
                    second[i] = instance.iAttr[indexSecond];
                }
                return correlation(dset, first, second);
            }
        }
    }
}
