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
package transformation;

import data.representation.DataSet;

/**
 * An interface for data transformers (like dimensionality reduction).
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface TransformationInterface {

    /**
     * @param dset DataSet object to transform.
     */
    public void setData(DataSet dset);

    /**
     * Performs data transformation.
     * 
     * @return DataSet object that is the transformed data. 
     */
    public DataSet transformData();
}
