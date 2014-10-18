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
package gui.images;

import javax.swing.JTextField;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import java.awt.FlowLayout;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

/**
 * This dialog sets up a choice for setting the minimal and maximal numbers of
 * clusters to use in clustering, prior to choosing the best configuration.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MinMaxClustersDialog extends JDialog implements ActionListener {

    // Input fields.
    private JTextField minClustersField;
    private JTextField maxClustersField;
    private JButton okButton;
    // The parent frame.
    private ImageManipulator parent;

    /**
     * Initialization.
     *
     * @param parentFrame ImageManipulator object that is the parent frame.
     */
    public MinMaxClustersDialog(ImageManipulator parentFrame) {
        super(parentFrame, "Choose min degree and weight", true);
        parent = parentFrame;
        setResizable(true);
        setSize(300, 130);
        getContentPane().setLayout(new FlowLayout());
        setModal(true);
        JLabel minClustersLabel = new JLabel("Min Number Of Clusters");
        JLabel maxClustersLabel = new JLabel("Max Number Of Clusters");
        minClustersField = new JTextField(6);
        maxClustersField = new JTextField(6);
        okButton = new JButton("Cluster");
        okButton.addActionListener(this);
        getContentPane().add(minClustersLabel);
        getContentPane().add(minClustersField);
        getContentPane().add(maxClustersLabel);
        getContentPane().add(maxClustersField);
        getContentPane().add(okButton);
        setVisible(true);
    }

    /**
     * Construct and show the dialog.
     *
     * @param parentFrame ImageManipulator object that is the parent frame.
     */
    public static void showDialog(ImageManipulator parentFrame) {
        MinMaxClustersDialog d = new MinMaxClustersDialog(parentFrame);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        int minClusters = 1;
        int maxClusters = 1;
        try {
            minClusters = Integer.parseInt(minClustersField.getText());
            maxClusters = Integer.parseInt(maxClustersField.getText());
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
        // Set the values for the minimal and maximal numbers of clusters to try
        // that the user has entered.
        parent.setForClustering(minClusters, maxClusters);
        dispose();
    }
}
