จาก Data set ที่ได้ พบว่ารูปแบบของ Data set คือ Exclusive OR dataset ซึ่งเป็นรูปแบบ 2 dimentions ที่แยกกลุ่มคลาสข้อมูลออกเป็นกลุ่ม 2 จุดด้วยกัน

โดยเริ่มแรกเราควร Normalize data ให้อยู่ในช่วงเดียวกันก่อนเพื่อลดปัญหา Bias ของ Model

จากนั้นก็ทำการการออกแบบ Model โดยเรานั้นสามารถใช้ Layer และจำนวน Node ที่ไม่มากในการสร้างได้เนื่องจากเราไม่ต้องการเส้นที่พับกันหลายชั้นเพื่อให้ได้เส้นแบ่งที่โค้ง (More layers) ขนาดนั้นเพราะ แค่สามารถแบ่งเส้นออกเป็น 2 กลุ่มก็สามารถแยก Class ได้ประสิทธิภาพสูงกว่า 70% แล้ว ผมจึงเลือกการออกแบบ Layer แค่ 3 Layer โดยแบ่งเป็นชั้นละ 4, 2, 1 Nodes เท่านั้น และกำหนดให้ใช้ Activation Function เป็น Relu เนื่องจาก
สามารถแยก Data set แบบนี้ได้ดี และปิดด้วย sigmoid

อีกทั้งในตอนสร้าง Model ผมได้มีการกำหนดการแบ่ง test size ให้เป็น 0.5 เพื่อป้องกันปัญหา Over-fitting และ Batch size เป็นทีละ 10 จากข้อมูล 100 และรอบการเทรน (epochs) เป็น 100 เพื่อเพิ่มความละเอียดในการเทรนใน Layer ที่น้อยและลดปัญหา Under-fitting เช่นกัน