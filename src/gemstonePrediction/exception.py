import sys 

def error_msg_detail(error, error_detail: sys):
  _, _, traceback = error_detail.exc_info()
  filename = traceback.tb_frame.f_code.co_filename
  line_number = traceback.tb_lineno
  error_msg = f"Error: {error} in {filename} at line {line_number}"

  return error_msg

class CustomException(Exception):
  def __init__(self, error_msg, error_detail: sys):
    super().__init__(error_msg)
    self.error_msg = error_msg_detail(error_msg, error_detail)

  def __str__(self):
    return self.error_msg