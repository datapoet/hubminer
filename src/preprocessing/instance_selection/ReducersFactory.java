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
package preprocessing.instance_selection;

/**
 * This class implements the methods for fetching initial configurations for
 * different instance selectors.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ReducersFactory {
    
    public static InstanceSelector getReducerForName(String name) {
        InstanceSelector selector;
        switch(name.toLowerCase()) {
            case "random":
            case "preprocessing.instance_selection.randomselector":
                selector = new RandomSelector();
                break;
            case "insight":
            case "insight.gh":
            case "preprocessing.instance_selection.insight":
                selector = new INSIGHT(null, INSIGHT.GOOD_HUBNESS);
                break;
            case "insight.gbh":
                selector = new INSIGHT(null,
                        INSIGHT.GOOD_MINUS_BAD_HUBNESS_PROP);
                break;
            case "insihgt.ghrel":
                selector = new INSIGHT(null,
                        INSIGHT.GOOD_HUBNESS_RELATIVE);
                break;
            case "insight.xi":
                selector = new INSIGHT(null, INSIGHT.XI);
                break;
            case "enn":
            case "preprocessing.instance_selection.wilson72":
                selector = new Wilson72();
                break;
            case "ipt_rt3":
            case "preprocessing.instance_selection.ipt_rt3":
                selector = new IPT_RT3();
                break;
            case "rnnr_al1":
            case "preprocessing.instance_selection.rnnr_al1":
                selector = new RNNR_AL1();
                break;
            case "cnn":
            case "preprocessing.instance_selection.cnn":
                selector = new CNN();
                break;
            case "gcnn":
            case "preprocessing.instance_selection.gcnn":
                selector = new GCNN(0.99f);
                break;
            case "gcnn01":
                selector = new GCNN(0.1f);
                break;
            default:
                selector = new RandomSelector();
                break;
        }
        return selector;
    }
    
}
