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
package gui.synthetic;

import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JTextField;

/**
 * This component implements the functionality for inserting a number of
 * instances from a Gaussian distribution into the panel, based on user input
 * parameters.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class InsertGaussianDialog extends JDialog implements ActionListener {

    private JTextField xSigmaField;
    private JTextField ySigmaField;
    private JTextField rotAngleField;
    private JTextField numInstField;
    private JButton okButton;
    private Visual2DdataGenerator parent;
    private float x, y;

    /**
     * Initialization.
     *
     * @param parentFrame Visual2DdataGenerator frame that called this dialog.
     * @param x Float value that is the current X coordinate.
     * @param y Float value that is the current Y coordinate.
     */
    public InsertGaussianDialog(Visual2DdataGenerator parentFrame, float x,
            float y) {
        super(parentFrame, "Choose distribution parameters", true);
        this.x = x;
        this.y = y;
        parent = parentFrame;
        setResizable(true);
        setSize(650, 130);
        getContentPane().setLayout(new FlowLayout());

        setModal(true);

        // Standard deviations input.
        JLabel xSLabel = new JLabel("X sigma:");
        JLabel ySLabel = new JLabel("Y sigma:");
        xSigmaField = new JTextField(6);
        ySigmaField = new JTextField(6);
        // The number of instances and the rotation angle.
        JLabel rotLabel = new JLabel("rotation angle(deg):");
        JLabel numLabel = new JLabel("num instances:");
        rotAngleField = new JTextField(6);
        numInstField = new JTextField(6);
        okButton = new JButton("Generate");
        okButton.addActionListener(this);

        // Insert all the components.
        getContentPane().add(xSLabel);
        getContentPane().add(xSigmaField);
        getContentPane().add(ySLabel);
        getContentPane().add(ySigmaField);
        getContentPane().add(rotLabel);
        getContentPane().add(rotAngleField);
        getContentPane().add(numLabel);
        getContentPane().add(numInstField);
        getContentPane().add(okButton);

        setVisible(true);
    }

    /**
     * Creates and shows a new Gaussian input dialog.
     *
     * @param parentFrame Visual2DdataGenerator frame that the request is for.
     * @param x Float value that is the current X coordinate.
     * @param y Float value that is the current Y coordinate.
     */
    public static void showDialog(Visual2DdataGenerator parentFrame,
            float x, float y) {
        InsertGaussianDialog d = new InsertGaussianDialog(parentFrame, x, y);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        float xSigma = 0.1f;
        float ySigma = 0.1f;
        float rotAngle = 0;
        int numInstances = 1;
        try {
            // Parse all the input fields.
            xSigma = Float.parseFloat(xSigmaField.getText());
            ySigma = Float.parseFloat(ySigmaField.getText());
            rotAngle = Float.parseFloat(rotAngleField.getText());
            numInstances = Integer.parseInt(numInstField.getText());
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
        // Generate and insert the requested points into the model.
        parent.insertGaussian(x, y, xSigma, ySigma, rotAngle, numInstances);
        dispose();
    }
}
