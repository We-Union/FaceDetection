from face_detection.utils import *

def test_png():
    img = Image.open("./test.png")
    img = np.array(img)
    print(img.shape)

def test_label():
    img_path = "./data/2002/08/11/big/img_591.jpg"
    x = 269.693400 
    y = 161.781200
    a = 123.583300
    b = 85.549500
    img = Image.open(img_path)
    img = np.array(img)
    plt.imshow(img)
    x1, y1, x2, y2 = ellipse_to_Rectangle_label(x, y, a, b)
    draw_one_boundingbox(x1, y1, x2, y2, color="c-", width=5, alpha=0.9)
    plt.show()

if __name__ == "__main__":
    with open("./data/meta.json", "r", encoding="utf-8") as fp:
        a : dict = json.load(fp=fp)
    path_list = list(a.keys())
    img_path = path_list[1]
    visualise_one_sample(img_path, a[img_path])