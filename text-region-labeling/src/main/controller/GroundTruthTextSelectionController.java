package main.controller;

import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextArea;
import javafx.scene.input.MouseEvent;
import javafx.stage.Stage;

import javax.annotation.PostConstruct;
public class GroundTruthTextSelectionController {
    @FXML
    private Button okBtn;
    @FXML
    private Button cancelBtn;
    @FXML
    private TextArea paragraphTxt;
    @FXML
    private TextArea selectedTxt;
    @FXML
    private ScrollPane scrollPane1;
    @FXML
    private ScrollPane scrollPane2;

    private StringBuilder outputString;
    private StringBuilder groundTruth;

    @PostConstruct
    public void initialize() {
        paragraphTxt.prefHeightProperty().bind(scrollPane1.heightProperty());
        paragraphTxt.prefWidthProperty().bind(scrollPane1.widthProperty());
        selectedTxt.prefHeightProperty().bind(scrollPane2.heightProperty());
        selectedTxt.prefWidthProperty().bind(scrollPane2.widthProperty());

        paragraphTxt.setOnMouseClicked(this::handleTextSelection);
        okBtn.setOnMouseClicked(this::handleOkButton);
        cancelBtn.setOnMouseClicked(this::handleCancelButton);
    }

    private void handleTextSelection(MouseEvent event) {
        selectedTxt.setText(paragraphTxt.getSelectedText());
    }

    public void setProperty(StringBuilder output, StringBuilder groundTruthTxt) {
        this.outputString = output;
        this.groundTruth = groundTruthTxt;
        paragraphTxt.setText(this.groundTruth.toString());
    }

    private void handleOkButton(MouseEvent event) {
        this.outputString.append(selectedTxt.getText().trim());
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Information Dialog");
        alert.setContentText("You have successfully added a text line.");
        alert.showAndWait();
        closeWindows();
    }

    private void handleCancelButton(MouseEvent event) {
        closeWindows();
    }

    private void closeWindows() {
        Stage stage = (Stage) okBtn.getScene().getWindow();
        stage.close();
    }
}