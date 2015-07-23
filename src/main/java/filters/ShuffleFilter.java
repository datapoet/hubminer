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
package filters;

import data.representation.DataInstance;
import data.representation.DataSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import util.AuxSort;

/**
 * Shuffles the original data.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ShuffleFilter implements FilterInterface {

    @Override
    public DataSet filterAndCopy(DataSet dset) {
        DataSet result = null;
        try {
            result = dset.copy();
            result.copy();
            filter(result);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        return result;
    }

    @Override
    public void filter(DataSet dset) {
        if (!dset.hasIdentifiers()) {
            // In this case we don't have to worry about two datasets, the main
            // one and the associated identifiers.
            Collections.shuffle(dset.data);
        } else {
            Random randa = new Random();
            float[] orderer = new float[dset.size()];
            for (int i = 0; i < orderer.length; i++) {
                orderer[i] = randa.nextFloat();
            }
            int[] indexes = null;
            try {
                indexes = AuxSort.sortIndexedValue(orderer, true);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
            DataSet identifiersDS = dset.identifiers;
            ArrayList<DataInstance> newInstances =
                    new ArrayList<>(dset.data.size());
            ArrayList<DataInstance> newIDs =
                    new ArrayList<>(dset.data.size());
            for (int i = 0; i < dset.size(); i++) {
                newInstances.add(dset.getInstance(indexes[i]));
                newIDs.add(identifiersDS.getInstance(indexes[i]));
            }
            dset.data = newInstances;
            identifiersDS.data = newIDs;
        }
    }
}
