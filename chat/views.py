import openai
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import HttpResponse
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from .serializers import ChatSerializer
from supabase import create_client, Client
#from transformers import TFAutoModel, AutoTokenizer
from transformers import LongformerTokenizer, TFLongformerModel
import numpy as np
import nltk
from nltk import sent_tokenize
nltk.download("punkt")

openai.api_key = "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"
SUPABASE_URL="https://xmfuzxgxaqmfwfzphfuk.supabase.co"
SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhtZnV6eGd4YXFtZndmenBoZnVrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjIxMjk0OTMsImV4cCI6MjAzNzcwNTQ5M30.TWRjlbLczM3t48dmn80OQH8yH628jx46q3rwo1ds_cc"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = TFLongformerModel.from_pretrained("allenai/longformer-base-4096")

def separar_texto(texto):
    return sent_tokenize(texto)

def vectorizar_texto(texto):
    oraciones = separar_texto(texto)
    vectores = []

    for oracion in oraciones:
        inputs = tokenizer(oracion, return_tensors="tf", max_length=4096, truncation=True)
        outputs = model(inputs)
        vectores.append(outputs.last_hidden_state.numpy().mean(axis=1).tolist()[0])

    vector_combinado = np.mean(vectores, axis=0)
    return vector_combinado

def almacenar_prompt(ruc, prompt, supabase):
    vector = vectorizar_texto(prompt).tolist()
    data = {
        "ruc": ruc,
        "prompt": prompt,
        "vector": vector
    }
    supabase.table("vectors").insert(data).execute()

def recuperar_vectores_prompt(ruc, supabase):
    response = supabase.table("vectors").select("*").eq("ruc", ruc).execute()
    return response.data

def similitud_coseno(vector1, vector2):
    v1 = np.array(vector1, dtype=float)
    v2 = np.array(vector2, dtype=float)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def encontrar_vectores_similares(nuevo_vector, vectores_almacenados, umbral=0.8):
    textos_similares = []
    for item in vectores_almacenados:
        vector_almacenado = item['vector']
        similitud = similitud_coseno(nuevo_vector, vector_almacenado)
        if similitud >= umbral:
            textos_similares.append(item['prompt'])
    return textos_similares

@api_view(['POST'])
@csrf_exempt
def read_jrxml_file(request, ruc):
    serializer = ChatSerializer(data=request.data)
    if serializer.is_valid():
        BASE_DIR =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        REPORTS_REFERENCE = os.path.join(BASE_DIR, 'Chat_reference', 'DE_FAC1.jrxml')
        with open(REPORTS_REFERENCE, 'r', encoding='utf-8') as file:
            jrxml_content = file.read()
            
        DE_FAC1_content = jrxml_content
        data = serializer.validated_data
        prompt = data['prompt']       
        # Almacenar el vector del prompt en Supabase usando el RUC
        almacenar_prompt(ruc, prompt, supabase)

        # Recuperar vectores previos si es necesario para contexto
        vectores_almacenados = recuperar_vectores_prompt(ruc, supabase)
        nuevo_vector = vectorizar_texto(prompt)
        textos_similares = encontrar_vectores_similares(nuevo_vector, vectores_almacenados)
        #mensaje_contexto = " ".join(textos_similares)

        parrafos = separar_texto(prompt)
        reference_data = {
            "#El contexto principal será el {DE_FAC1_content}.",
            "#No añadirás ningún parámetro, field o variable al jrxml de referencia, siempre seguirás la misma estructura.",
            "#Devolverás como respuesta algunas guías de como añadir algo solicitado y luego la sección del jrxml solicitado por el usuario, no hagas toda la estructura del jrxml solo la pequeña porción de código.",
            "#Puedes indicar algunas opciones para poder modificar o añadir algo en el reporte y luego con la estructura del código jrxml de referencia: {DE_FAC1_content} y la solicitud del usuario sin afectar la estructura general."
        }
        for parrafo in parrafos:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Eres un experto programador en generación de reportes en Jaspersoft Studio 6.19, siempre respondes en español. Tu referencia será el siguiente archivo JRXML: {DE_FAC1_content} y más detalles con {reference_data}"},
                    {"role": "user", "content": parrafo}
                ],
                stream=True
            )
        recolectar_mensaje = []
        for chunk in response:
            if "choices" in chunk:
                for choice in chunk["choices"]:
                    if "delta" in choice:
                        if "content" in choice["delta"]:
                            chunk_messages = choice["delta"]["content"]
                            recolectar_mensaje.append(chunk_messages)
        combined_message = recolectar_mensaje
        return HttpResponse(combined_message, content_type="text/plain")
            
    return Response({'message': 'Error al validar datos, verificarlo'}, status=status.HTTP_400_BAD_REQUEST)