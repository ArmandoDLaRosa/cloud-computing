#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

int main() {
    // -------------------------------------------------
    // VECTORES
    // -------------------------------------------------
    constexpr size_t N = 50'000'000; // Un número grande, pero puedes ajustarlo según tu sistema
    std::vector<float> a(N), b(N), c_serial(N), c_parallel(N);

    // -------------------------------------------------
    // DATOS
    // -------------------------------------------------
    for (size_t i = 0; i < N; ++i) {
        a[i] = i * 1.5f;
        b[i] = i + 3.7f;
    }

    // -------------------------------------------------
    // FUNCIONES
    // -------------------------------------------------

    // Funciones para reiniciar el vector 'c_parallel'  y para 
    // medir tiempos con una directiva de OpenMP concreta.
    auto limpiarC = [&](){
        for (size_t i = 0; i < N; ++i) {
            c_parallel[i] = 0.0f;
        }
    };

    auto medirTiempo = [&](auto f) {
        auto inicio = std::chrono::high_resolution_clock::now();
        f();
        auto fin = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(fin - inicio).count();
    };

    // ----------------------------------------------------------
    // SERIAL
    // ----------------------------------------------------------
    auto inicio_serial = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        c_serial[i] = a[i] + b[i];
    }
    auto fin_serial = std::chrono::high_resolution_clock::now();
    double tiempo_serial = std::chrono::duration<double>(fin_serial - inicio_serial).count();
    
    // ----------------------------------------------------------
    // OPEN MP
    // ----------------------------------------------------------

    std::vector<int> chunks = {100, 10000};

    double tiempo_static_default  = 0.0;
    double tiempo_dynamic_default = 0.0;
    std::vector<double> tiempo_static_ch, tiempo_dynamic_ch;
    tiempo_static_ch.resize(chunks.size());
    tiempo_dynamic_ch.resize(chunks.size());

    //  schedule(static) por defecto
    limpiarC();
    tiempo_static_default = medirTiempo([&]() {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            c_parallel[i] = a[i] + b[i];
        }
    });

    // schedule(dynamic) por defecto
    limpiarC();
    tiempo_dynamic_default = medirTiempo([&]() {
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < N; ++i) {
            c_parallel[i] = a[i] + b[i];
        }
    });

    // schedule(static, chunk) con varios chunks
    for (size_t idx = 0; idx < chunks.size(); ++idx) {
        int chunkSize = chunks[idx];
        limpiarC();
        if (chunkSize > 0) {
            tiempo_static_ch[idx] = medirTiempo([&]() {
                #pragma omp parallel for schedule(static, chunkSize)
                for (size_t i = 0; i < N; ++i) {
                    c_parallel[i] = a[i] + b[i];
                }
            });
        } else {
            tiempo_static_ch[idx] = medirTiempo([&]() {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < N; ++i) {
                    c_parallel[i] = a[i] + b[i];
                }
            });
        }
    }

    // schedule(dynamic, chunk) con varios chunks
    for (size_t idx = 0; idx < chunks.size(); ++idx) {
        int chunkSize = chunks[idx];
        limpiarC();
        if (chunkSize > 0) {
            tiempo_dynamic_ch[idx] = medirTiempo([&]() {
                #pragma omp parallel for schedule(dynamic, chunkSize)
                for (size_t i = 0; i < N; ++i) {
                    c_parallel[i] = a[i] + b[i];
                }
            });
        } else {
            tiempo_dynamic_ch[idx] = medirTiempo([&]() {
                #pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < N; ++i) {
                    c_parallel[i] = a[i] + b[i];
                }
            });
        }
    }

    // -------------------------------------------------
    // RESULTADOS
    // -------------------------------------------------

    std::cout << "\nRESULTADOS DEL PROGRAMA\n"
              << "-----------------------\n\n";

    std::cout << "Primeros 5 valores de 'a': ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << "\nPrimeros 5 valores de 'b': ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << b[i] << " ";
    }

    std::cout << "\nPrimeros 5 valores (ejecución SERIAL) de 'c_serial': ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << c_serial[i] << " ";
    }
    std::cout << "\n\n";

    // Tiempos ----------------------------------------------------------------

    // Serial
    std::cout << "Tiempo (Suma SERIAL): " << tiempo_serial << " segundos\n\n";

    // Paralelo - statico
    std::cout << "[PARALELO] schedule(static) default  -> " << tiempo_static_default  << " seg\n";
    for (size_t idx = 0; idx < chunks.size(); ++idx) {
        int chunkSize = chunks[idx];
        std::cout << "[PARALELO] schedule(static, " 
                  << (chunkSize > 0 ? std::to_string(chunkSize) : "default") 
                  << ") -> " << tiempo_static_ch[idx] << " seg\n";
    }
    std::cout << "\n";

    // Paralelo - dinamico
    std::cout << "[PARALELO] schedule(dynamic) default -> " << tiempo_dynamic_default << " seg\n";
    for (size_t idx = 0; idx < chunks.size(); ++idx) {
        int chunkSize = chunks[idx];
        std::cout << "[PARALELO] schedule(dynamic, " 
                  << (chunkSize > 0 ? std::to_string(chunkSize) : "default") 
                  << ") -> " << tiempo_dynamic_ch[idx] << " seg\n";
    }
    std::cout << "\n";

    return 0;
}
