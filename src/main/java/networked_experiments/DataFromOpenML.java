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
package networked_experiments;

import data.representation.DataSet;
import java.util.ArrayList;

/**
 * This object holds the data obtained from the OpenML task for use in
 * classification experiments. This includes the dataset and the folds.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DataFromOpenML {
    
    // Currently public for simplicity, since there are many fields to consider.
    // This class is more like a record, it does not implement any
    // functionality.
    public int numFolds;
    public int numTimes;
    public String[] classNames;
    public DataSet filteredDSet;
    public ArrayList<Integer>[][][] trainTestIndexes;
    public String targetFeatureName;
}
