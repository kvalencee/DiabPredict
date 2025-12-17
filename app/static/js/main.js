/**
 * DiabPredict - JavaScript Principal
 */

// Inicialización cuando el DOM está listo
document.addEventListener('DOMContentLoaded', function() {
    console.log('DiabPredict initialized');

    // Inicializar todos los tooltips
    initializeTooltips();

    // Inicializar validación de formularios
    initializeFormValidation();
});

/**
 * Inicializa todos los tooltips de Bootstrap
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Inicializa la validación de formularios
 */
function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');

    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
}

/**
 * Formatea un número con separadores de miles
 * @param {number} num - Número a formatear
 * @returns {string} - Número formateado
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Formatea una fecha a formato legible
 * @param {string} dateString - String de fecha ISO
 * @returns {string} - Fecha formateada
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    const options = {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    return date.toLocaleDateString('es-ES', options);
}

/**
 * Obtiene el color según el nivel de riesgo
 * @param {string} riskLevel - Nivel de riesgo ('Bajo', 'Medio', 'Alto')
 * @returns {string} - Clase CSS de color
 */
function getRiskColor(riskLevel) {
    const colors = {
        'Bajo': 'success',
        'Medio': 'warning',
        'Alto': 'danger'
    };
    return colors[riskLevel] || 'secondary';
}

/**
 * Muestra un mensaje de notificación
 * @param {string} message - Mensaje a mostrar
 * @param {string} type - Tipo de alerta ('success', 'danger', 'warning', 'info')
 */
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.setAttribute('role', 'alert');
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);

        // Auto-cerrar después de 5 segundos
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
}

/**
 * Realiza una petición API
 * @param {string} url - URL de la API
 * @param {object} options - Opciones de fetch
 * @returns {Promise} - Promesa con la respuesta
 */
async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API Request Error:', error);
        throw error;
    }
}

/**
 * Carga datos del historial
 * @returns {Promise<Array>} - Array de evaluaciones
 */
async function loadEvaluations() {
    try {
        const data = await apiRequest('/api/evaluations');
        return data.evaluations || [];
    } catch (error) {
        console.error('Error loading evaluations:', error);
        showNotification('Error al cargar el historial', 'danger');
        return [];
    }
}

/**
 * Elimina una evaluación
 * @param {string} evaluationId - ID de la evaluación
 * @returns {Promise<boolean>} - true si se eliminó correctamente
 */
async function deleteEvaluation(evaluationId) {
    if (!confirm('¿Está seguro de que desea eliminar esta evaluación?')) {
        return false;
    }

    try {
        await apiRequest(`/api/evaluations/${evaluationId}`, {
            method: 'DELETE'
        });
        showNotification('Evaluación eliminada exitosamente', 'success');
        return true;
    } catch (error) {
        console.error('Error deleting evaluation:', error);
        showNotification('Error al eliminar la evaluación', 'danger');
        return false;
    }
}

/**
 * Exporta resultados a texto
 * @param {object} evaluation - Datos de la evaluación
 */
function exportResults(evaluation) {
    const text = `
DIABPREDICT - RESULTADOS DE EVALUACIÓN
========================================

ID de Evaluación: ${evaluation.id}
Fecha: ${formatDate(evaluation.timestamp)}

PARÁMETROS CLÍNICOS:
--------------------
Embarazos: ${evaluation.parameters.pregnancies}
Glucosa: ${evaluation.parameters.glucose} mg/dL
Presión Arterial: ${evaluation.parameters.blood_pressure} mmHg
Grosor Pliegue Cutáneo: ${evaluation.parameters.skin_thickness} mm
Insulina: ${evaluation.parameters.insulin} µU/mL
IMC: ${evaluation.parameters.bmi} kg/m²
Función Pedigree: ${evaluation.parameters.pedigree_function}
Edad: ${evaluation.parameters.age} años

RESULTADO:
----------
Nivel de Riesgo: ${evaluation.result.risk_level}
Probabilidad: ${evaluation.result.probability}%

RECOMENDACIONES:
----------------
${evaluation.result.recommendations.join('\n\n')}

========================================
IMPORTANTE: Este resultado es una estimación y NO constituye un diagnóstico médico.
Consulte con un profesional de la salud calificado.
    `.trim();

    // Crear blob y descargar
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `diabpredict_${evaluation.id}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Copia texto al portapapeles
 * @param {string} text - Texto a copiar
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copiado al portapapeles', 'success');
    }).catch(err => {
        console.error('Error copying to clipboard:', err);
        showNotification('Error al copiar al portapapeles', 'danger');
    });
}

/**
 * Imprime la página actual
 */
function printPage() {
    window.print();
}

// Exponer funciones globalmente
window.DiabPredict = {
    showNotification,
    apiRequest,
    loadEvaluations,
    deleteEvaluation,
    exportResults,
    copyToClipboard,
    printPage,
    formatDate,
    formatNumber,
    getRiskColor
};