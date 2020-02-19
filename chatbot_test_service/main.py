from service import app

if __name__ == "__main__":
    # host = "https://musedev.iptime.org
    host = "0.0.0.0"
    port = "9500" 
    
    app.run(host=host, port=port)
