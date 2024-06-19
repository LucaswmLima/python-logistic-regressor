from PyQt5.QtWidgets import QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout, QMessageBox, QFileDialog, QLineEdit, QHBoxLayout, QGridLayout, QDesktopWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from styled_button import StyledButton
from help_dialog import HelpDialog
from regression_logic import RegressionLogic

class RegressionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.regression_logic = RegressionLogic()

    def initUI(self):
        self.setWindowTitle('Regressão Logística')
        self.resize(800, 600)
        self.setStyleSheet("background-color: white;")
        self.setWindowIcon(QIcon('assets/icon.png'))

        layout = QGridLayout()
        layout.setSpacing(10)

        self.separator_label = QLabel('Separador:', self)
        layout.addWidget(self.separator_label, 0, 0)
        self.separator_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.separator_combobox = QComboBox(self)
        self.separator_combobox.addItems([',', ';', ':', '|', '/'])
        layout.addWidget(self.separator_combobox, 0, 1)

        self.load_button = StyledButton('Carregar CSV', self)
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button, 0, 2)

        self.column_x_label = QLabel('Coluna de X:', self)
        layout.addWidget(self.column_x_label, 1, 0)
        self.column_x_combobox = QComboBox(self)
        self.column_x_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.column_x_combobox, 1, 1)

        self.column_y_label = QLabel('Coluna de Y:', self)
        layout.addWidget(self.column_y_label, 2, 0)
        layout.setColumnStretch(1, 1)
        self.column_y_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.column_y_combobox = QComboBox(self)
        layout.addWidget(self.column_y_combobox, 2, 1)

        self.accuracy_label = QLabel('', self)
        layout.addWidget(self.accuracy_label, 2, 2, Qt.AlignCenter)

        self.confusion_matrix_label = QLabel('', self)
        layout.addWidget(self.confusion_matrix_label, 3, 0, 1, 3, Qt.AlignCenter)

        self.run_regression_button = StyledButton('Rodar Regressão', self)
        self.run_regression_button.setEnabled(False)
        self.run_regression_button.clicked.connect(self.run_regression)
        layout.addWidget(self.run_regression_button, 5, 0)

        self.plot_regression_button = StyledButton('Mostrar Regressão', self)
        self.plot_regression_button.setEnabled(False)
        self.plot_regression_button.clicked.connect(self.plot_regression)
        layout.addWidget(self.plot_regression_button, 5, 1)

        hbox = QHBoxLayout()
        hbox.setSpacing(10)

        self.input_label = QLabel('Valor de X:', self)
        self.input_label.setStyleSheet('QLabel {font-size: 12pt;}')
        hbox.addWidget(self.input_label)

        self.input_value = QLineEdit(self)
        self.input_value.setStyleSheet('QLineEdit {font-size: 12pt;}')
        hbox.addWidget(self.input_value)

        self.predict_button = StyledButton('Prever', self)
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self.predict)
        hbox.addWidget(self.predict_button)

        layout.addLayout(hbox, 6, 0, 1, 3)

        self.help_button = QPushButton('Ajuda', self)
        self.help_button.setStyleSheet("""
            QPushButton {
                background-color: #cc092f;
                color: white;
                font-size: 12pt;
                min-width: 20px;
                max-width: 50px;
                height: 20px;
                border-radius: 15px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #e60032;
            }
            QPushButton:pressed {
                background-color: #b30026;
            }
        """)
        self.help_button.clicked.connect(self.show_help_dialog)
        layout.addWidget(self.help_button, 7, 0, 1, 2)

        self.setLayout(layout)

        self.file_path = None
        self.data = None
        self.model = None
        self.X = None
        self.y = None

        self.column_x_combobox.currentIndexChanged.connect(self.update_metrics)
        self.column_y_combobox.currentIndexChanged.connect(self.update_metrics)

    def load_csv(self):
        separator = self.separator_combobox.currentText()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Selecionar Arquivo CSV', '', 'CSV Files (*.csv)')
        if file_path:
            try:
                self.data = self.regression_logic.load_data(file_path, separator)
                self.column_x_combobox.clear()
                self.column_y_combobox.clear()
                self.column_x_combobox.addItems(self.data.columns)
                self.column_y_combobox.addItems(self.data.columns)
                QMessageBox.information(self, "Informação", "CSV carregado com sucesso!")
                self.run_regression_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao carregar CSV: {e}")

    def run_regression(self):
        if self.data is not None:
            try:
                x_column = self.column_x_combobox.currentText()
                y_column = self.column_y_combobox.currentText()
                self.regression_logic.run_regression(x_column, y_column)
                self.plot_regression_button.setEnabled(True)
                self.predict_button.setEnabled(True)
                self.update_metrics()
                QMessageBox.information(self, "Informação", "Regressão rodada com sucesso!")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao rodar regressão: {e}")

    def update_metrics(self):
        if self.column_x_combobox.currentText() and self.column_y_combobox.currentText():
            x_column = self.column_x_combobox.currentText()
            y_column = self.column_y_combobox.currentText()
            try:
                self.regression_logic.run_regression(x_column, y_column)
                if self.regression_logic.accuracy is not None:
                    self.accuracy_label.setText(f"Acurácia: {self.regression_logic.accuracy:.2f}")
                else:
                    self.accuracy_label.setText("")

                if self.regression_logic.confusion_matrix is not None:
                    cm_text = f"Matriz de Confusão:\n{self.regression_logic.confusion_matrix}"
                    self.confusion_matrix_label.setText(cm_text)
                else:
                    self.confusion_matrix_label.setText("")
            except Exception as e:
                self.accuracy_label.setText("")
                self.confusion_matrix_label.setText("")
                print(f"Erro ao atualizar métricas: {e}")

    def plot_regression(self):
        if self.regression_logic.model is not None:
            self.regression_logic.plot_regression()

    def predict(self):
        try:
            x_value = float(self.input_value.text())
            y_pred = self.regression_logic.predict(x_value)
            QMessageBox.information(self, "Previsão", f"O valor previsto de Y para X={x_value} é: {y_pred[0]}")
        except ValueError:
            QMessageBox.critical(self, "Erro", "Por favor, insira um valor numérico válido para X.")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao prever valor: {e}")

    def show_help_dialog(self):
        help_dialog = HelpDialog(self)
        help_dialog.exec_()

    def center_window(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

