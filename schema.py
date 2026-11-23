from pydantic import BaseModel, Field
from typing import Literal, Optional

class Parte(BaseModel):
    nombre: str = Field(description="Nombre completo de la persona física o razón social de la empresa que actúa como esta parte del contrato")
    representante: Optional[str] = Field(default=None, description="Nombre del representante legal o administrador que firma en nombre de la empresa. Debe ser None si la parte es una persona física que actúa en nombre propio")

class DatosContrato(BaseModel):
    tipo_contrato: Literal["Local de Negocio", "Vivienda", "Plaza de Garaje"] = Field(description="identificar el tipo de contrato al que pertenece")
    ciudad: str = Field(description="Ciudad donde se ubica el inmueble arrendado (no la ciudad de firma si difieren)")
    arrendador: Parte
    arrendatario: Parte
    direccion_inmueble: str = Field(description="Dirección completa del inmueble arrendado, incluyendo calle, número, código postal y ciudad")

class RutaConsulta(BaseModel):
    razon: str = Field(description="Justificación breve de la ruta elegida (obligatorio)")
    ruta: Literal["TABLA", "CLAUSULAS", "AMBAS"] = Field(
        description=(
            "Fuente necesaria para responder la pregunta. "
            "TABLA: la respuesta se obtiene de los datos de identificación de los contratos "
            "(tipo de contrato, ciudad, nombres de arrendador y arrendatario, representantes, "
            "dirección del inmueble), o requiere contar, comparar o listar contratos completos. "
            "CLAUSULAS: la respuesta está en el texto del contrato (importes, rentas, fianzas, "
            "plazos, duración, obligaciones, penalizaciones, seguros, anexos, condiciones). "
            "AMBAS: hace falta identificar uno o varios contratos por sus datos y además "
            "consultar su contenido."
        )
    )


