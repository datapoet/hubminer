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
package visualization;

/**
 * This enumeration contains the chart types for viper charts.
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public enum ChartType {
        PR_SPACE, ROC_SPACE, PR_CURVES, LIFT_CURVES, ROC_CURVES,
        ROC_HULL_CURVES, COST_CURVES, RATE_DRIVEN_CURVES, KENDALL_CURVES,
        COLUMN_CHART
}
