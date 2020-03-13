package main;

import com.sun.javafx.stage.StageHelper;
import javafx.application.Platform;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.scene.control.TextArea;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.logging.Level;
import java.util.logging.Logger;

public class App extends javafx.application.Application {

    public static void main(String[] args) {
        launch(args);
    }

    public static void showExceptionAlert(Exception e) {
        Logger.getAnonymousLogger().log(Level.SEVERE, e.getMessage(), e);
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setHeaderText(e.getClass().getName());

        StringWriter stringWriter = new StringWriter();
        PrintWriter printWriter = new PrintWriter(stringWriter);
        e.printStackTrace(printWriter);
        TextArea area = new TextArea(stringWriter.toString());
        area.setWrapText(true);
        area.setEditable(false);

        alert.getDialogPane().setContent(area);
        alert.setResizable(true);
        alert.show();
    }

    @Override
    public void start(Stage browserStage) throws Exception {
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("browser.fxml"));
        Parent loader = fxmlLoader.load();
        BrowserController browserController = fxmlLoader.getController();
        browserController.setOwnerStage(browserStage);

        browserStage.setScene(new Scene(loader, 800, 120));
        browserStage.setTitle("Image Labeling");
        browserStage.centerOnScreen();
        browserStage.setResizable(false);
        browserStage.show();
        browserStage.setOnCloseRequest(event -> {
            if (StageHelper.getStages().size() > 1) {
                showClosingWarning(event);
                event.consume();
            }
        });
    }

    private void showClosingWarning(WindowEvent event) {
        Alert alert = new Alert(Alert.AlertType.WARNING,
                "This window is the only way to select the output file. Close this window will close" +
                        " the entire application. Select YES to force close the application or NO to just minimize",
                ButtonType.YES, ButtonType.NO, ButtonType.CANCEL);
        alert.initModality(Modality.WINDOW_MODAL);
        alert.showAndWait().ifPresent(buttonType -> {
            if (buttonType == ButtonType.YES) {
                Platform.exit();
            } else if (buttonType == ButtonType.NO) {
                ((Stage) event.getSource()).setIconified(true);
            }
        });
    }
}
