import logging
import json
import cv2
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import VideoFrame
import numpy as np


import yt_dlp

import process_video as pv

logger = logging.getLogger("server")

app = FastAPI()


# Configuração CORS para permitir todas as origens (apenas para desenvolvimento)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


CATTLE_COUNTING=0
CATTLE_WEIGHT=0

#video = "./videos/video_puro.mp4"
video = "https://www.youtube.com/watch?app=desktop&time_continue=12&v=ZPfvxijOg80&embeds_referring_euri=http%3A%2F%2Fgadopesado.com%2F&source_ve_path=MjM4NTE&feature=emb_title"


# Variáveis globais para armazenar a posição atual do vídeo
last_frame_timestamp = None
last_frame_time_base = None
current_frame_number = 0
video_lock = asyncio.Lock()





class VideoTransformTrack(VideoStreamTrack):

    def __init__(self, video):
        super().__init__()
        self.video = video
        self.cap = cv2.VideoCapture(video)
        self.pv = pv.ProcessVideo(video)
        self.frame_skip = 2

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video}")


    async def recv(self):
        global current_frame_number, last_frame_timestamp, last_frame_time_base
        try:
            async with video_lock:
                # Define o frame atual
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)

                # Lê o próximo frame do vídeo
                ret, frame = self.cap.read()
                if ret:
                    # Processa o frame se for necessário
                    if current_frame_number % self.frame_skip == 0:
                        frame, cattle_counting, cattle_weight = self.pv.start(frame, show_analyze=False)
                    else:
                        frame, cattle_counting, cattle_weight = self.pv.show(frame)

                    # Atualiza as variáveis globais de contagem e peso do gado
                    global CATTLE_COUNTING, CATTLE_WEIGHT
                    CATTLE_COUNTING = cattle_counting
                    CATTLE_WEIGHT = cattle_weight

                    # Converte o frame para RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Cria um VideoFrame a partir do array do frame
                    frame = VideoFrame.from_ndarray(frame, format="rgb24")
                    # Define o timestamp do frame
                    if last_frame_timestamp is not None and last_frame_time_base is not None:
                        frame.pts = last_frame_timestamp + 1
                        frame.time_base = last_frame_time_base
                    else:
                        frame.pts = 0
                        frame.time_base = 1

                    # Atualiza o timestamp do último frame enviado
                    last_frame_timestamp = frame.pts
                    last_frame_time_base = frame.time_base

                    # Incrementa o contador de frames
                    current_frame_number += 1
                    return frame
                else:
                    return None

        except Exception as e:
            print("An unexpected error occurred recv:", e)
            return None



# Função para obter a URL do fluxo de vídeo
def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'bestvideo',
        'quiet': True,
        'noplaylist': True,
        'extract_flat': 'in_playlist',
        'skip_download': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict.get('url')
    return video_url



# Instância global da classe VideoTransformTrack
if video.startswith("http"):
    video_stream_url = get_youtube_stream_url(video)
    video_track = VideoTransformTrack(video_stream_url)
else:
    video_track = VideoTransformTrack(video)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pc = RTCPeerConnection()

    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            await websocket.send_text(json.dumps({
                "type": "candidate",
                "candidate": candidate.toJSON()
            }))
            print("Sent ICE candidate to client")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed":
            await pc.close()
            await websocket.close()

    try:
        print('WebSocket connection established')
        pc.addTrack(video_track)
        print("Added video track to peer connection")

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await websocket.send_text(json.dumps({
            "type": "offer",
            "sdp": pc.localDescription.sdp,
            "sdp_type": pc.localDescription.type  # Enviar sdp_type
        }))
        print("Sent offer to client")

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message["type"] == "answer":
                await pc.setRemoteDescription(RTCSessionDescription(
                    sdp=message["sdp"],
                    type=message["sdp_type"]
                ))
            elif message["type"] == "candidate":
                candidate = message["candidate"]
                await pc.addIceCandidate(candidate)
            else:
                print(f"Unknown message type: {message['type']}")


    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"Error during WebSocket communication: {e}")
    finally:
        await pc.close()


# Inicia o processo de envio de informação do gado
###################################################


# Lista de clientes WebSocket conectados
websocket_clients = []

# Função para enviar o número atualizado para todos os clientes WebSocket
async def send_number_to_clients():
    print('send_number_to_clients')
    while True:
        # Criar uma mensagem para enviar para os clientes
        message = {"type": "cattle_info", "amount": CATTLE_COUNTING, "weight": CATTLE_WEIGHT}
        # Enviar a mensagem para todos os clientes conectados
        for client in websocket_clients:
            try:
                await client.send_text(json.dumps(message))
            except Exception as e:
                print(f"Error send cattle info client: {e}")
                websocket_clients.remove(client)
        # Aguardar um intervalo de tempo (por exemplo, 3 segundos) antes de envia para o client
        await asyncio.sleep(3)

# Iniciar a tarefa de envio de número para clientes
async def start_number_update_task():
    print('start_number_update_task ')
    await send_number_to_clients()

# Iniciar a tarefa de envio de número para clientes quando o servidor é iniciado
@app.on_event("startup")
async def startup_event():
    print('startup_event ')
    asyncio.create_task(start_number_update_task())

# Rota para lidar com conexões WebSocket
@app.websocket("/ws_info")
async def websocket_info(websocket: WebSocket):
    # Aceitar a conexão WebSocket
    await websocket.accept()
    # Adicionar o cliente à lista de clientes WebSocket conectados
    websocket_clients.append(websocket)
    try:
        # Manter a conexão WebSocket aberta indefinidamente
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("WebSocket disconnected Info.")
        websocket_clients.remove(websocket)
    except Exception as e:
        print(f"Error during WebSocket Info communication: {e}")
    finally:
        websocket_clients.remove(client)


def clean():

    # Limpando as variáveis globais de contagem e peso do gado
    global CATTLE_COUNTING, CATTLE_WEIGHT, current_frame_number
    CATTLE_COUNTING = 0
    CATTLE_WEIGHT = 0
    current_frame_number = 0

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
