/**
 * JavaScript port of the multilateration library
 * Based on the original Rust implementation
 */

/**
 * Represents a point in n-dimensional space
 */
class Point {
    constructor(coordinates) {
        this.coordinates = coordinates;
    }
}

/**
 * Represents a measurement with a known point and distance
 */
class Measurement {
    constructor(point, distance) {
        this.point = point;
        this.distance = distance;
    }
}

/**
 * Validates that all measurements have the same dimensions
 * @param {Array<Measurement>} measurements - Array of measurements
 * @throws {Error} If validation fails
 */
function validateMeasurements(measurements) {
    if (!measurements || measurements.length === 0) {
        throw new Error("Measurements cannot be empty");
    }

    const pointDimensions = measurements.map(m => m.point.coordinates.length);
    const minLength = Math.min(...pointDimensions);
    const maxLength = Math.max(...pointDimensions);

    if (minLength !== maxLength) {
        throw new Error("All points must have the same dimensions");
    }
    if (minLength < 1) {
        throw new Error("Points must contain at least one dimension");
    }
}

/**
 * Function to estimate an initial point from measurements
 * @param {Array<Measurement>} measurements - Array of measurements
 * @returns {Point} Initial estimated point
 */
function estimateInitialPoint(measurements) {
    const positionDimensions = measurements[0].point.coordinates.length;
    const numberOfMeasurements = measurements.length;

    const initialPosition = new Array(positionDimensions).fill(0);

    for (let i = 0; i < numberOfMeasurements; i++) {
        for (let j = 0; j < positionDimensions; j++) {
            initialPosition[j] += measurements[i].point.coordinates[j];
        }
    }

    for (let i = 0; i < positionDimensions; i++) {
        initialPosition[i] /= numberOfMeasurements;
    }

    return new Point(initialPosition);
}

/**
 * Calculates the error function for a given position
 * @param {Array<number>} position - Current position estimate
 * @param {Array<Measurement>} measurements - Array of measurements
 * @returns {Array<number>} Array of errors
 */
function calculateErrors(position, measurements) {
    return measurements.map(measurement => {
        let squaredDistance = 0;
        for (let j = 0; j < position.length; j++) {
            squaredDistance += Math.pow(position[j] - measurement.point.coordinates[j], 2);
        }
        return squaredDistance - Math.pow(measurement.distance, 2);
    });
}

/**
 * Calculates the Jacobian matrix for the optimization
 * @param {Array<number>} position - Current position estimate
 * @param {Array<Measurement>} measurements - Array of measurements
 * @returns {Array<Array<number>>} Jacobian matrix
 */
function calculateJacobian(position, measurements) {
    const matrix = [];

    for (let i = 0; i < measurements.length; i++) {
        const row = [];
        for (let j = 0; j < position.length; j++) {
            row.push(2 * position[j] - 2 * measurements[i].point.coordinates[j]);
        }
        matrix.push(row);
    }

    return matrix;
}

/**
 * Transposes a matrix
 * @param {Array<Array<number>>} matrix - Input matrix
 * @returns {Array<Array<number>>} Transposed matrix
 */
function transposeMatrix(matrix) {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result = Array(cols).fill().map(() => Array(rows).fill(0));

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

/**
 * Multiplies two matrices
 * @param {Array<Array<number>>} a - First matrix
 * @param {Array<Array<number>>} b - Second matrix
 * @returns {Array<Array<number>>} Result matrix
 */
function multiplyMatrices(a, b) {
    const rowsA = a.length;
    const colsA = a[0].length;
    const colsB = b[0].length;
    const result = Array(rowsA).fill().map(() => Array(colsB).fill(0));

    for (let i = 0; i < rowsA; i++) {
        for (let j = 0; j < colsB; j++) {
            let sum = 0;
            for (let k = 0; k < colsA; k++) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

/**
 * Multiplies a matrix by a vector
 * @param {Array<Array<number>>} matrix - Input matrix
 * @param {Array<number>} vector - Input vector
 * @returns {Array<number>} Result vector
 */
function multiplyMatrixVector(matrix, vector) {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result = new Array(rows).fill(0);

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    return result;
}

/**
 * Adds lambda to the diagonal of a matrix
 * @param {Array<Array<number>>} matrix - Input matrix
 * @param {number} lambda - Value to add to diagonal
 * @returns {Array<Array<number>>} Modified matrix
 */
function addLambdaToDiagonal(matrix, lambda) {
    const result = matrix.map(row => [...row]);
    const size = Math.min(result.length, result[0].length);

    for (let i = 0; i < size; i++) {
        result[i][i] += lambda;
    }

    return result;
}

/**
 * Solves a linear system using Gaussian elimination
 * @param {Array<Array<number>>} A - Coefficient matrix
 * @param {Array<number>} b - Right-hand side vector
 * @returns {Array<number>} Solution vector
 */
function solveLinearSystem(A, b) {
    const n = A.length;
    const augmentedMatrix = A.map((row, i) => [...row, b[i]]);

    // Gaussian elimination (forward elimination)
    for (let i = 0; i < n; i++) {
        // Find pivot
        let maxRow = i;
        for (let j = i + 1; j < n; j++) {
            if (Math.abs(augmentedMatrix[j][i]) > Math.abs(augmentedMatrix[maxRow][i])) {
                maxRow = j;
            }
        }

        // Swap rows
        if (maxRow !== i) {
            [augmentedMatrix[i], augmentedMatrix[maxRow]] = [augmentedMatrix[maxRow], augmentedMatrix[i]];
        }

        // Singular matrix check
        if (Math.abs(augmentedMatrix[i][i]) < 1e-10) {
            throw new Error("Matrix is singular, cannot solve system");
        }

        // Eliminate below
        for (let j = i + 1; j < n; j++) {
            const factor = augmentedMatrix[j][i] / augmentedMatrix[i][i];
            for (let k = i; k <= n; k++) {
                augmentedMatrix[j][k] -= factor * augmentedMatrix[i][k];
            }
        }
    }

    // Back substitution
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        x[i] = augmentedMatrix[i][n];
        for (let j = i + 1; j < n; j++) {
            x[i] -= augmentedMatrix[i][j] * x[j];
        }
        x[i] /= augmentedMatrix[i][i];
    }

    return x;
}

/**
 * Subtracts two vectors
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {Array<number>} Result vector
 */
function subtractVectors(a, b) {
    return a.map((val, i) => val - b[i]);
}

/**
 * Calculates the dot product of two vectors
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {number} Dot product
 */
function dotProduct(a, b) {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

/**
 * Performs multilateration to find a point based on distance measurements
 * @param {Array<Measurement>} measurements - Array of measurements
 * @returns {Point} The calculated position
 * @throws {Error} If calculation fails
 */
function multilaterate(measurements) {
    validateMeasurements(measurements);

    const initialPoint = estimateInitialPoint(measurements);
    let currentPosition = [...initialPoint.coordinates];

    const maxIterations = 1000;
    const initialLambda = 0.01;
    const lambdaFactor = 10;

    let lambda = initialLambda;
    let lastError = Number.MAX_VALUE;

    for (let iteration = 0; iteration < maxIterations; iteration++) {
        // Calculate current error values
        const errors = calculateErrors(currentPosition, measurements);
        const currentError = Math.sqrt(errors.reduce((sum, err) => sum + err * err, 0));

        // Check if we're done
        if (currentError < 1e-10) {
            return new Point(currentPosition);
        }

        // If error increased, increase lambda and try again
        if (currentError > lastError) {
            lambda *= lambdaFactor;
            continue;
        }

        // Calculate Jacobian
        const jacobian = calculateJacobian(currentPosition, measurements);
        const jacobianT = transposeMatrix(jacobian);

        // Calculate JTJ + lambda*I
        const JTJ = multiplyMatrices(jacobianT, jacobian);
        const damped = addLambdaToDiagonal(JTJ, lambda);

        // Calculate JTr
        const JTr = multiplyMatrixVector(jacobianT, errors);

        try {
            // Solve (JTJ + lambda*I) * delta = JTr
            const delta = solveLinearSystem(damped, JTr);

            // Update position
            currentPosition = subtractVectors(currentPosition, delta);

            // Update lambda
            lambda /= lambdaFactor;
            lastError = currentError;
        } catch (error) {
            lambda *= lambdaFactor;
        }
    }

    return new Point(currentPosition);
}

// Export the public API
module.exports = {
    Point,
    Measurement,
    multilaterate
};
