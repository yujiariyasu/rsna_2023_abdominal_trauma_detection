import gspread
from oauth2client.service_account import ServiceAccountCredentials

class GoogleSpreadSheetEditor:
    def __init__(self, title):
        try:
            for i in range(10):
                scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
                credentials = ServiceAccountCredentials.from_json_keyfile_name('/home/acc12347av/ml_pipeline/src/utils/gspread.json', scope)
                gc = gspread.authorize(credentials)
                self.wks = gc.open(title).sheet1
                self.fold_col_map = {0: 10, 1:11, 2:12, 3:13, 4:14}
                self.fold_0_col = 10
                break
        except Exception as e:
            print('='*100, 'GoogleSpreadSheetEditor error!!!')
            print(e)
            print('='*100, 'GoogleSpreadSheetEditor error!!!')
            pass
